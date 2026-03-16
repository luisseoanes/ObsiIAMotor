#include "../include/obsia_api.h"
#include "llm_engine.h"
#include "rag_engine.h"
#include "clinical_engine.h"
#include "text_norm.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <mutex>
#include <unordered_set>
#include <cctype>
#include <thread>
#include <fstream>

// Auto-detect optimal thread count: all cores minus 1 (leave 1 for OS/UI)
static int detect_optimal_threads() {
    int cores = 0;
#ifdef __ANDROID__
    // On Android, read /sys/devices/system/cpu/possible for accurate count
    std::ifstream f("/sys/devices/system/cpu/possible");
    if (f.is_open()) {
        // Format is "0-N" where N+1 is the number of cores
        std::string line;
        std::getline(f, line);
        auto dash = line.find('-');
        if (dash != std::string::npos) {
            cores = std::stoi(line.substr(dash + 1)) + 1;
        }
    }
#endif
    if (cores <= 0) {
        cores = (int)std::thread::hardware_concurrency();
    }
    if (cores <= 0) cores = 4; // Safe fallback
    int optimal = cores - 1;
    return optimal >= 2 ? optimal : 2; // Minimum 2 threads
}

static std::unique_ptr<LLMEngine> g_llm_engine;
static std::unique_ptr<RAGEngine> g_rag_engine;
static ClinicalEngine g_clinical_engine;
static std::mutex g_mutex;
static int g_rag_k = 3;
static bool g_low_quality_model = false;

// Keep thread local or static for the C API return
static std::string g_last_response;

static const char* SYSTEM_PROMPT =
    "Eres ObsIA, un medico obstetra experto. "
    "Responde en espanol con claridad, rigor y tono profesional. "
    "Usa exclusivamente el contexto RAG provisto en el mensaje del usuario; no inventes ni completes con conocimientos externos. "
    "Solo puedes agregar preguntas de aclaracion y advertencias de seguridad basadas en los sintomas del usuario. "
    "Si el contexto es insuficiente, indicalo y pide los datos clinicos faltantes. "
    "Evita terminos incompatibles con el estado obstetrico. "
    "Si hay signos de alarma, recomienda atencion inmediata. "
    "Responde en 3 a 5 frases. ";

static std::string truncate_text(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    return s.substr(0, max_len) + "...";
}

static std::string join_items(const std::vector<std::string>& items, size_t max_items, const char* sep) {
    std::ostringstream oss;
    size_t added = 0;
    for (const auto& it : items) {
        if (it.empty()) continue;
        if (added > 0) oss << sep;
        oss << truncate_text(it, 120);
        added++;
        if (added >= max_items) break;
    }
    return oss.str();
}

static std::vector<std::string> tokenize_spanish(const std::string& s) {
    static const std::unordered_set<std::string> STOP = {
        "a","ante","bajo","con","de","desde","el","la","los","las",
        "un","una","unos","unas","y","o","u","en","por","para","que",
        "si","su","sus","al","del","lo","como","segun","mas",
        "sin","es","son","fue","fueron","ser","esta","estan",
        "se","me","mi","mis","tu","tus","le","les","ya","hay"
    };

    std::vector<std::string> out;
    std::string cur;
    std::string norm = normalize_spanish_lower(s);
    for (size_t i = 0; i <= norm.size(); ++i) {
        unsigned char c = (i < norm.size()) ? (unsigned char)norm[i] : ' ';
        bool is_word = std::isalnum(c) || static_cast<signed char>(c) < 0;
        if (is_word) {
            cur.push_back((char)std::tolower(c));
        } else if (!cur.empty()) {
            if (STOP.find(cur) == STOP.end() && cur.size() >= 3) {
                out.push_back(cur);
            }
            cur.clear();
        }
    }
    return out;
}

static int count_token_overlap(const std::string& a, const std::string& b) {
    auto ta = tokenize_spanish(a);
    auto tb = tokenize_spanish(b);
    if (ta.empty() || tb.empty()) return 0;
    std::unordered_set<std::string> sb(tb.begin(), tb.end());
    int hits = 0;
    for (const auto& t : ta) {
        if (sb.find(t) != sb.end()) {
            hits++;
            if (hits >= 3) break;
        }
    }
    return hits;
}

static bool is_low_quality_model_path(const char* path) {
    if (!path) return false;
    std::string p = normalize_spanish_lower(std::string(path));
    if (p.find("iq2") != std::string::npos) return true;
    if (p.find("q2_k") != std::string::npos) return true;
    if (p.find("q2k") != std::string::npos) return true;
    if (p.find("q2") != std::string::npos) return true;
    return false;
}

static bool contains_context_dump(const std::string& s) {
    if (s.find("###") != std::string::npos) return true;
    if (s.find("guias medicas") != std::string::npos || s.find("GUIAS MEDICAS") != std::string::npos) return true;
    int bullets = 0;
    std::istringstream iss(s);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind("- ", 0) == 0) bullets++;
        if (bullets >= 2) return true;
    }
    return false;
}

static bool is_gibberish(const std::string& s) {
    if (s.size() < 20) return true;
    std::string norm = normalize_spanish_lower(s);
    int letters = 0, digits = 0, other = 0;
    int words = 0, long_words = 0, no_vowel_words = 0;
    std::string cur;

    auto flush_word = [&]() {
        if (cur.empty()) return;
        words++;
        if (cur.size() >= 18) long_words++;
        int vowels = 0;
        for (char c : cur) {
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') vowels++;
        }
        if (cur.size() >= 4 && vowels == 0) no_vowel_words++;
        cur.clear();
    };

    for (char c : norm) {
        unsigned char uc = (unsigned char)c;
        if (std::isalpha(uc) || static_cast<signed char>(c) < 0) {
            letters++;
            cur.push_back(c);
        } else if (std::isdigit(uc)) {
            digits++;
            cur.push_back(c);
        } else if (std::isspace(uc)) {
            flush_word();
        } else {
            other++;
        }
    }
    flush_word();

    int non_space = letters + digits + other;
    if (non_space == 0) return true;
    double letter_ratio = (double)letters / (double)non_space;
    if (letter_ratio < 0.65) return true;
    if (words >= 6 && long_words >= 2) return true;
    if (words >= 6 && no_vowel_words * 3 > words) return true;
    return false;
}

static std::string pick_urgent_warning(const std::vector<std::string>& warnings) {
    for (const auto& w : warnings) {
        std::string lw = normalize_spanish_lower(w);
        if (lw.find("emergencia") != std::string::npos ||
            lw.find("atencion inmediata") != std::string::npos ||
            lw.find("urgente") != std::string::npos ||
            lw.find("sangrado") != std::string::npos ||
            lw.find("convulsion") != std::string::npos ||
            lw.find("severo") != std::string::npos) {
            return w;
        }
    }
    return "";
}

static bool validate_response(const std::string& response, const ClinicalPlan& plan, const std::string& rag_context) {
    if (response.size() < 20) return false;
    const std::string out = normalize_spanish_lower(response);
    const std::string rag = normalize_spanish_lower(rag_context);
    for (const auto& term : plan.banned_terms) {
        if (term.empty()) continue;
        if (out.find(term) != std::string::npos) return false;
    }
    int required_total = 0;
    int required_hits = 0;
    for (const auto& term : plan.required_keywords) {
        if (term.empty()) continue;
        if (rag.find(term) != std::string::npos) {
            required_total++;
            if (out.find(term) != std::string::npos) required_hits++;
        }
    }
    if (required_total > 0 && required_hits > 0) return true;
    if (!plan.evidence.empty()) {
        for (const auto& ev : plan.evidence) {
            if (count_token_overlap(response, ev) >= 2) return true;
        }
    }
    if (!rag_context.empty() && count_token_overlap(response, rag_context) >= 2) return true;
    return false;
}

static std::string extract_first_bullet(const std::string& rag_context) {
    std::istringstream iss(rag_context);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind("- ", 0) != 0) continue;
        std::string txt = line.substr(2);
        return txt;
    }
    return "";
}

static std::vector<std::string> split_sentences(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        cur.push_back(c);
        if (c == '.' || c == '!' || c == '?') {
            // Avoid splitting on abbreviations like "Dr."
            if (cur.size() >= 2 && std::isupper((unsigned char)cur[0]) && cur.size() <= 4) {
                continue;
            }
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static std::string trim_sentences(const std::string& s, size_t max_sent) {
    auto parts = split_sentences(s);
    if (parts.size() <= max_sent) return s;
    std::ostringstream oss;
    for (size_t i = 0; i < max_sent; ++i) {
        if (i > 0) oss << " ";
        oss << parts[i];
    }
    return oss.str();
}

static std::string build_structured_response(const ClinicalPlan& plan, const std::string& rag_context) {
    std::ostringstream oss;
    if (!plan.diagnoses.empty()) {
        oss << "Con el contexto disponible, se considera " << plan.diagnoses[0] << ".";
    } else {
        oss << "Con el contexto disponible, no es posible dar un diagnostico definitivo.";
    }

    std::string ev;
    if (!plan.evidence.empty()) {
        ev = truncate_text(plan.evidence[0], 180);
    } else {
        ev = extract_first_bullet(rag_context);
        ev = truncate_text(ev, 180);
    }
    if (!ev.empty()) {
        oss << " Del contexto: " << ev;
        if (!ev.empty() && ev.back() != '.') oss << ".";
    }

    if (!plan.questions.empty()) {
        oss << " Para orientar mejor, necesito: " << join_items(plan.questions, 2, "; ") << ".";
    }

    std::string urgent = pick_urgent_warning(plan.warnings);
    if (!urgent.empty()) {
        if (!urgent.empty() && urgent.back() != '.') urgent += ".";
        oss << " " << urgent;
    }

    std::string out = oss.str();
    out = trim_sentences(out, 5);
    auto parts = split_sentences(out);
    if (parts.size() < 3) {
        out += " Si hay empeoramiento, busque atencion inmediata.";
        out = trim_sentences(out, 5);
    }
    return out;
}

static std::string build_safe_response(const ClinicalPlan& plan, const std::string& rag_context, bool rag_missing) {
    std::ostringstream oss;
    if (rag_missing) {
        oss << "No se encontro contexto clinico en la base RAG para tu consulta. ";
    } else {
        oss << "Segun el contexto disponible, ";
    }
    if (!rag_context.empty()) {
        if (!plan.evidence.empty()) {
            const size_t max_ev = std::min<size_t>(2, plan.evidence.size());
            for (size_t i = 0; i < max_ev; ++i) {
                if (i > 0) oss << " ";
                oss << truncate_text(plan.evidence[i], 180);
                if (!plan.evidence[i].empty() && plan.evidence[i].back() != '.') oss << ".";
            }
        } else {
            std::string bullet = extract_first_bullet(rag_context);
            if (!bullet.empty()) {
                oss << truncate_text(bullet, 180);
                if (!bullet.empty() && bullet.back() != '.') oss << ".";
            } else {
                oss << "el contexto no aporta detalles clinicos suficientes. ";
            }
        }
    }
    if (!plan.questions.empty()) {
        oss << " Para orientar mejor, necesito: " << join_items(plan.questions, 2, "; ") << ".";
    } else if (!rag_missing) {
        oss << " Puedes indicar sintomas y datos clinicos adicionales?";
    }
    std::string urgent = pick_urgent_warning(plan.warnings);
    if (!urgent.empty()) {
        if (!urgent.empty() && urgent.back() != '.') urgent += ".";
        oss << " " << urgent;
    }
    std::string out = oss.str();
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
        out.pop_back();
    }
    return out;
}

static void obsia_free_unlocked() {
    g_rag_engine.reset();
    g_llm_engine.reset();
    g_low_quality_model = false;
    g_last_response.clear();
}

extern "C" {

int obsia_init(const ObsiaConfig* config) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!config || !config->model_path) return -1;

    obsia_free_unlocked();

    g_low_quality_model = is_low_quality_model_path(config->model_path);
    g_rag_k = config->rag_k > 0 ? config->rag_k : 3;
    g_llm_engine = std::make_unique<LLMEngine>();
    int n_ctx = config->n_ctx > 0 ? config->n_ctx : 1024;
    int n_batch = config->n_batch > 0 ? config->n_batch : 128;
    int n_threads = config->n_threads > 0 ? config->n_threads : detect_optimal_threads();

    if (!g_llm_engine->load_model(config->model_path, n_ctx, n_batch, n_threads)) {
        g_llm_engine.reset();
        return -2; // Default error for LLM load fail
    }

    if (config->rag_chunks_path) {
        try {
            g_rag_engine = std::make_unique<RAGEngine>(config->rag_chunks_path);
        } catch (...) {
            g_llm_engine.reset();
            return -3; // Default error for RAG load fail
        }
    }

    return 0;
}

void obsia_free() {
    std::lock_guard<std::mutex> lock(g_mutex);
    obsia_free_unlocked();
}

int obsia_is_ready() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_llm_engine ? 1 : 0;
}

const char* obsia_chat(const char* user_message) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_llm_engine || !user_message) return "Error: Engine no inicializado.";

    std::string user(user_message);
    if (user.empty()) return "";

    ObstetricState state = g_clinical_engine.detect_state(user);

    std::string rag_context;
    int rag_k = g_rag_k > 0 ? g_rag_k : 3;
    if (g_low_quality_model && rag_k > 2) rag_k = 2;
    if (g_rag_engine) {
        rag_context = g_rag_engine->get_augmented_context(user, rag_k, state);
    }

    ClinicalPlan plan = g_clinical_engine.build_plan(user, rag_context);

    if (rag_context.empty()) {
        g_last_response = build_safe_response(plan, rag_context, true);
        return g_last_response.c_str();
    }

    std::string final_user_msg;
    if (g_low_quality_model) {
        final_user_msg =
            "Contexto RAG:\n" + rag_context +
            "\nResponde en 3 a 5 frases, solo con este contexto. "
            "Menciona 1 o 2 conceptos del contexto. "
            "Si falta informacion, pregunta datos clinicos. "
            "Si hay signos de alarma, recomienda atencion inmediata.\n";
    } else {
        final_user_msg =
            "Contexto RAG (usa solo esto):\n" + rag_context +
            "\nInstrucciones:\n"
            "- Responde en 3 a 5 frases.\n"
            "- Usa solo informacion del Contexto RAG.\n"
            "- Menciona explicitamente 1 o 2 conceptos clave presentes en el Contexto RAG.\n"
            "- Si falta informacion, indicalo y haz preguntas concretas.\n"
            "- Si hay signos de alarma descritos por el usuario, recomienda atencion inmediata.\n";
    }

    if (!plan.banned_terms.empty()) {
        size_t banned_limit = g_low_quality_model ? 4 : 6;
        final_user_msg += "- No menciones: " + join_items(plan.banned_terms, banned_limit, ", ") + ".\n";
    }

    final_user_msg += "Consulta del usuario:\n" + user;

    const char* tmpl = llama_model_chat_template(g_llm_engine->get_model(), nullptr);
    if (!tmpl) {
        g_last_response = "Error: Template de chat no disponible.";
        return g_last_response.c_str();
    }

    std::vector<llama_chat_message> msg_ptrs;
    msg_ptrs.push_back({"system", SYSTEM_PROMPT});
    msg_ptrs.push_back({"user", final_user_msg.c_str()});
    
    std::vector<char> formatted(llama_n_ctx(g_llm_engine->get_context()));
    int new_len = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), true, formatted.data(), (int)formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, msg_ptrs.data(), (int)msg_ptrs.size(), true, formatted.data(), (int)formatted.size());
    }
    if (new_len <= 0) {
        g_last_response = "Error: No se pudo formatear el prompt.";
        return g_last_response.c_str();
    }

    std::string final_formatted_prompt(formatted.begin(), formatted.begin() + new_len);

    std::string model_reply = g_llm_engine->generate_response(final_formatted_prompt);
    bool ok = validate_response(model_reply, plan, rag_context);
    if (ok && (contains_context_dump(model_reply) || is_gibberish(model_reply))) {
        ok = false;
    }
    if (ok) {
        std::string out = trim_sentences(model_reply, 5);
        auto parts = split_sentences(out);
        if (parts.size() < 3) {
            if (!plan.questions.empty()) {
                out += " Para orientar mejor, necesito: " + join_items(plan.questions, 2, "; ") + ".";
            }
            std::string urgent = pick_urgent_warning(plan.warnings);
            if (!urgent.empty()) {
                if (!urgent.empty() && urgent.back() != '.') urgent += ".";
                out += " " + urgent;
            }
            out = trim_sentences(out, 5);
        }
        g_last_response = out;
    } else {
        g_last_response = build_structured_response(plan, rag_context);
    }
    return g_last_response.c_str();
}

} // extern "C"
