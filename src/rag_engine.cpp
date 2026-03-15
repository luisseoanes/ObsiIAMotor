#include "rag_engine.h"
#include "clinical_engine.h"
#include "text_norm.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "json.hpp"  // Local json.hpp

using json = nlohmann::json;

static std::string to_lower(const std::string& s) {
    return normalize_spanish_lower(s);
}

static bool contains_any(const std::string& haystack, const std::vector<std::string>& needles) {
    for (const auto& n : needles) {
        if (haystack.find(n) != std::string::npos) return true;
    }
    return false;
}

static int count_words_approx(const std::string& s) {
    int count = 0;
    bool in_word = false;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        bool is_word = std::isalnum(c) || static_cast<signed char>(c) < 0;
        if (is_word) {
            if (!in_word) {
                count++;
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    return count;
}

static std::string trim_to_words(const std::string& s, int max_words) {
    if (max_words <= 0) return "";
    int count = 0;
    bool in_word = false;
    size_t end = s.size();
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        bool is_word = std::isalnum(c) || static_cast<signed char>(c) < 0;
        if (is_word) {
            if (!in_word) {
                count++;
                if (count > max_words) {
                    end = i;
                    break;
                }
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    std::string out = s.substr(0, end);
    if (end < s.size()) {
        while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
            out.pop_back();
        }
        out += "...";
    }
    return out;
}

static bool label_is_useful(const std::string& label) {
    if (label.empty()) return false;
    std::string l = to_lower(label);
    if (l.size() < 3 || l.size() > 60) return false;
    return contains_any(l, {
        "hemorragia", "preeclampsia", "eclampsia", "sepsis", "ruptura",
        "prematuro", "placenta", "cesarea", "hipertension", "infeccion",
        "manejo", "tratamiento", "diagnost", "criter", "signos de alarma",
        "emergencia", "rpm"
    });
}

static std::string make_chunk_label(const Chunk& c) {
    std::string label;
    if (!c.clinical_topic.empty()) {
        label += c.clinical_topic;
    }
    if (!c.section_type.empty()) {
        if (!label.empty()) label += " | ";
        label += c.section_type;
    }
    if (label.empty()) return "";
    if (!label_is_useful(label)) return "";
    return "[" + label + "] ";
}

// Expande la consulta con tÃ©rminos clÃ­nicos para que BM25 recupere las guÃ­as correctas (p. ej. hemorragia posparto).
std::string RAGEngine::expand_query_for_obstetrics(const std::string& query) const {
    std::string qlower = to_lower(query);
    std::string out = query;

    if (contains_any(qlower, {"sangr", "hemorragia", "sangrado"}) &&
        contains_any(qlower, {"parir", "parto", "posparto", "postparto", "alumbramiento", "puerperio"})) {
        out += " hemorragia posparto hemorragia postparto atonia uterina oxitocina masaje uterino uterotonico manejo hemorragia";
    }
    if (contains_any(qlower, {"preeclampsia", "presion alta", "tension", "hipertension", "eclampsia"}) ||
        (contains_any(qlower, {"embarazada", "gestante"}) && contains_any(qlower, {"presion", "tension", "cefalea", "vision"}))) {
        out += " preeclampsia eclampsia criterios diagnosticos manejo sulfato magnesio signos alarma";
    }
    if (contains_any(qlower, {"sepsis", "fiebre", "escalofrios", "infeccion", "mal olor"})) {
        out += " sepsis obstetrica infeccion puerperal antibioticos signos alarma";
    }
    if (contains_any(qlower, {"bolsa rota", "ruptura de membranas", "rom", "liquido"})) {
        out += " ruptura prematura de membranas manejo rpm corioamnionitis";
    }
    if (contains_any(qlower, {"prematuro", "prematura", "preterm", "amenaza parto", "contracciones antes"})) {
        out += " parto prematuro amenaza parto prematuro manejo tocolisis corticoides";
    }
    if (contains_any(qlower, {"sangrado", "hemorragia"}) && contains_any(qlower, {"embarazo", "gestante"})) {
        out += " placenta previa desprendimiento de placenta dppni manejo";
    }
    if (contains_any(qlower, {"cesarea", "operacion", "cirugia"}) && contains_any(qlower, {"parto", "embarazo"})) {
        out += " cesarea indicaciones contraindicaciones";
    }

    return out;
}

std::string RAGEngine::detect_topic_hint(const std::string& query) const {
    const std::string q = to_lower(query);
    if (contains_any(q, {"hemorragia", "sangrado", "aton"})) return "hemorragia";
    if (contains_any(q, {"preeclampsia", "eclampsia", "hipertension"})) return "preeclampsia";
    if (contains_any(q, {"sepsis", "infeccion", "fiebre"})) return "sepsis";
    if (contains_any(q, {"rom", "ruptura de membranas", "bolsa rota"})) return "ruptura";
    if (contains_any(q, {"prematuro", "parto prematuro"})) return "prematuro";
    if (contains_any(q, {"placenta previa", "desprendimiento", "dppni"})) return "placenta";
    if (contains_any(q, {"cesarea"})) return "cesarea";
    return "";
}

bool RAGEngine::is_chunk_compatible(const Chunk& chunk, ObstetricState state) const {
    const std::string txt = to_lower(chunk.text);
    const std::string topic = to_lower(chunk.clinical_topic);

    if (state == ObstetricState::POSTPARTO) {
        if (contains_any(txt, {"feto", "fcf", "doppler", "viabilidad", "monitorizacion fetal", "cardiotocografia"})) return false;
        if (contains_any(topic, {"monitoreo fetal", "embarazo", "control prenatal"})) return false;
    }
    if (state == ObstetricState::ANTENATAL || state == ObstetricState::INTRAPARTO) {
        if (contains_any(txt, {"posparto", "postparto", "puerperio"})) return false;
    }
    return true;
}

static std::string dirname(const std::string& path) {
    size_t p = path.find_last_of("/\\");
    if (p == std::string::npos) return ".";
    return path.substr(0, p);
}

RAGEngine::RAGEngine(const std::string& chunks_path) {
    load_chunks(chunks_path);
    
    std::vector<std::string> texts;
    for (const auto& chunk : chunks_) {
        texts.push_back(chunk.text);
    }
    ranker_.index_documents(texts);
    
    load_embeddings_if_present(chunks_path);
}

void RAGEngine::load_embeddings_if_present(const std::string& chunks_path) {
    std::string base = dirname(chunks_path);
    std::string emb_path = base + "/embeddings.bin";
    std::ifstream f(emb_path, std::ios::binary);
    if (!f.is_open()) return;
    
    int32_t n = 0, dim = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (n <= 0 || dim <= 0 || n != (int)chunks_.size()) {
        return;
    }
    size_t need = (size_t)n * (size_t)dim;
    embeddings_.resize(need);
    f.read(reinterpret_cast<char*>(embeddings_.data()), need * sizeof(float));
    if (!f) return;
    embed_dim_ = dim;
    has_embeddings_ = true;
    printf("ðŸ“Š RAG hÃ­brido: embeddings cargados (%d vectores, dim=%d).\n", n, dim);
}

void RAGEngine::load_chunks(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir: " + path);
    }
    
    json j;
    file >> j;
    
    for (const auto& item : j) {
        Chunk chunk;
        if (item.contains("text")) {
             chunk.text = item["text"];
        }
        if (item.contains("metadata")) {
            if (item["metadata"].contains("source"))
                chunk.source = item["metadata"]["source"];
            if (item["metadata"].contains("chunk_id"))
                chunk.chunk_id = item["metadata"]["chunk_id"];
            if (item["metadata"].contains("word_count"))
                chunk.word_count = item["metadata"]["word_count"];
            if (item["metadata"].contains("section_type"))
                chunk.section_type = item["metadata"]["section_type"];
            if (item["metadata"].contains("clinical_topic"))
                chunk.clinical_topic = item["metadata"]["clinical_topic"];
        }
        chunks_.push_back(chunk);
    }
    printf("ðŸ“Š RAG: %zu fragmentos cargados de la base de conocimientos.\n", chunks_.size());
}

std::vector<std::string> RAGEngine::search(const std::string& query, int k) {
    auto results = ranker_.search(query, k);
    
    std::vector<std::string> relevant_texts;
    std::unordered_set<std::string> seen;
    
    for (const auto& result : results) {
        if (result.score > 0.0f) {
            const std::string& txt = chunks_[result.chunk_id].text;
            if (seen.find(txt) == seen.end()) {
                relevant_texts.push_back(txt);
                seen.insert(txt);
            }
        }
    }
    
    if ((int)relevant_texts.size() > k) {
        relevant_texts.resize(k);
    }
    
    return relevant_texts;
}

std::vector<int> RAGEngine::top_k_similar_to(int chunk_id, int k) const {
    if (!has_embeddings_ || chunk_id < 0 || (size_t)chunk_id >= chunks_.size()) return {};
    const float* v = embeddings_.data() + (size_t)chunk_id * (size_t)embed_dim_;
    int n = (int)chunks_.size();
    std::vector<std::pair<int, float>> scores;
    scores.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (i == chunk_id) continue;
        const float* u = embeddings_.data() + (size_t)i * (size_t)embed_dim_;
        float dot = 0.f, norm_v = 0.f, norm_u = 0.f;
        for (int d = 0; d < embed_dim_; ++d) {
            dot += v[d] * u[d];
            norm_v += v[d] * v[d];
            norm_u += u[d] * u[d];
        }
        float den = std::sqrt(norm_v) * std::sqrt(norm_u);
        float sim = (den > 1e-9f) ? (dot / den) : 0.f;
        scores.push_back({i, sim});
    }
    std::partial_sort(scores.begin(), scores.begin() + std::min(k, (int)scores.size()), scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<int> out;
    for (int i = 0; i < std::min(k, (int)scores.size()); ++i) {
        out.push_back(scores[i].first);
    }
    return out;
}

std::string RAGEngine::get_augmented_context(const std::string& query, int k, ObstetricState state) {
    const std::string search_query = expand_query_for_obstetrics(query);
    const int k_query = std::max(k * 4, k);
    auto bm25_results = ranker_.search(search_query, k_query);
    const std::string topic_hint = detect_topic_hint(query);

    struct Cand { int id; float score; };
    std::vector<Cand> candidates;
    candidates.reserve(bm25_results.size());

    for (const auto& r : bm25_results) {
        if (r.score <= 0.0f) continue;
        const Chunk& c = chunks_[r.chunk_id];
        if (!is_chunk_compatible(c, state)) continue;

        float score = r.score;
        const std::string section = to_lower(c.section_type);
        if (contains_any(section, {"manejo", "tratamiento", "algoritmo", "signos de alarma", "diagnostico", "criterios"})) {
            score *= 1.15f;
        }
        if (!topic_hint.empty()) {
            const std::string topic = to_lower(c.clinical_topic);
            if (topic.find(topic_hint) != std::string::npos) {
                score *= 1.20f;
            }
        }
        candidates.push_back({r.chunk_id, score});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
        return a.score > b.score;
    });

    std::unordered_set<int> seen_ids;
    for (const auto& c : candidates) {
        seen_ids.insert(c.id);
    }
    std::unordered_set<int> used_id;
    std::vector<int> ordered_ids;

    if (has_embeddings_ && !candidates.empty()) {
        int top1 = candidates[0].id;
        auto similar = top_k_similar_to(top1, 2);
        for (int idx : similar) {
            if (seen_ids.count(idx)) continue;
            if (!is_chunk_compatible(chunks_[idx], state)) continue;
            seen_ids.insert(idx);
            candidates.push_back({idx, candidates[0].score * 0.7f});
        }
        std::sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b) {
            return a.score > b.score;
        });
    }

    for (const auto& c : candidates) {
        if ((int)ordered_ids.size() >= k) break;
        if (used_id.count(c.id)) continue;
        used_id.insert(c.id);
        ordered_ids.push_back(c.id);
    }

    if (ordered_ids.empty()) return "";

    const size_t MAX_CHUNK_CHARS = 420;
    const int MAX_CHUNK_WORDS = 80;
    const int MAX_TOTAL_WORDS = 220;

    std::ostringstream oss;
    oss << "### GUIAS MEDICAS (Contexto):\n";
    int total_words = 0;
    for (size_t i = 0; i < ordered_ids.size(); ++i) {
        const Chunk& c = chunks_[ordered_ids[i]];
        std::string chunk = c.text;
        int words = c.word_count > 0 ? c.word_count : count_words_approx(chunk);
        if (words <= 0) words = 1;
        if (words > MAX_CHUNK_WORDS) {
            chunk = trim_to_words(chunk, MAX_CHUNK_WORDS);
        }
        if (chunk.size() > MAX_CHUNK_CHARS) {
            chunk = chunk.substr(0, MAX_CHUNK_CHARS) + "...";
        }
        int used_words = std::min(words, MAX_CHUNK_WORDS);
        if (total_words + used_words > MAX_TOTAL_WORDS) break;
        std::string label = make_chunk_label(c);
        oss << "- " << label << chunk << "\n";
        total_words += used_words;
    }
    return oss.str();
}


