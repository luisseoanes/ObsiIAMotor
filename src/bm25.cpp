#include "bm25.h"
#include "text_norm.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <cctype>

BM25Ranker::BM25Ranker(float k1, float b) : k1_(k1), b_(b), avg_doc_length_(0), next_term_id_(0) {}

std::vector<std::string> BM25Ranker::tokenize(const std::string& text) {
    static const std::unordered_set<std::string> STOPWORDS = {
        "a", "ante", "bajo", "con", "de", "desde", "el", "la", "los", "las", 
        "un", "una", "unos", "unas", "y", "o", "u", "en", "por", "para", 
        "que", "si", "su", "sus", "al", "del", "lo", "recomienda",
        "recomendacion", "guia", "experticia", "experiencia", "clinica",
        "paciente", "mujer", "caso", "casos"
    };

    std::vector<std::string> tokens;
    std::string current;
    
    const std::string norm = normalize_spanish_lower(text);
    for (size_t i = 0; i <= norm.length(); ++i) {
        char c = (i < norm.length()) ? norm[i] : ' ';
        
        // Caracteres alfanuméricos (incluyendo UTF-8 multi-byte que son negativos)
        if (std::isalnum(static_cast<unsigned char>(c)) || static_cast<signed char>(c) < 0) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!current.empty()) {
            if (STOPWORDS.find(current) == STOPWORDS.end()) {
                tokens.push_back(current);
            }
            current.clear();
        }
    }
    
    return tokens;
}

int BM25Ranker::get_term_id(const std::string& term, bool create_if_missing) {
    auto it = vocab_.find(term);
    if (it != vocab_.end()) {
        return it->second;
    }
    
    if (create_if_missing) {
        int id = next_term_id_++;
        vocab_[term] = id;
        return id;
    }
    
    return -1;
}

void BM25Ranker::index_documents(const std::vector<std::string>& docs) {
    doc_term_freqs_.clear();
    doc_freqs_.clear();
    doc_lengths_.clear();
    vocab_.clear();
    next_term_id_ = 0;
    
    long long total_length = 0;
    
    for (const auto& doc : docs) {
        auto tokens = tokenize(doc);
        doc_lengths_.push_back((int)tokens.size());
        total_length += tokens.size();
        
        std::unordered_map<int, int> term_freq;
        std::unordered_set<int> unique_terms;
        
        for (const auto& token : tokens) {
            int term_id = get_term_id(token, true);
            term_freq[term_id]++;
            unique_terms.insert(term_id);
        }
        
        doc_term_freqs_.push_back(term_freq);
        
        for (int term_id : unique_terms) {
            doc_freqs_[term_id]++;
        }
    }
    
    if (!docs.empty()) {
        avg_doc_length_ = static_cast<float>(total_length) / docs.size();
    } else {
        avg_doc_length_ = 0;
    }
}

float BM25Ranker::compute_idf(int term_id) {
    float N = (float)doc_term_freqs_.size();
    float df = (float)(doc_freqs_.count(term_id) ? doc_freqs_[term_id] : 0);
    
    return std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);
}

std::vector<BM25Result> BM25Ranker::search(const std::string& query, int k) {
    auto query_tokens = tokenize(query);
    std::vector<BM25Result> results;
    
    // Pre-calcular IDs de query para no buscar en mapa repetidamente
    std::vector<int> query_term_ids;
    for (const auto& term : query_tokens) {
        int id = get_term_id(term, false);
        if (id != -1) {
            query_term_ids.push_back(id);
        }
    }

    results.reserve(doc_term_freqs_.size());
    
    for (size_t doc_id = 0; doc_id < doc_term_freqs_.size(); ++doc_id) {
        float score = 0.0f;
        
        for (int term_id : query_term_ids) {
            // Si el término no existe en el documento, skip
            if (doc_term_freqs_[doc_id].find(term_id) == doc_term_freqs_[doc_id].end()) continue;

            float idf = compute_idf(term_id);
            int tf = doc_term_freqs_[doc_id].at(term_id);
            
            float doc_len = (float)doc_lengths_[doc_id];
            float norm = 1.0f;
            if (avg_doc_length_ > 0.0f) {
                norm = 1.0f - b_ + b_ * (doc_len / avg_doc_length_);
            }
            
            score += idf * (tf * (k1_ + 1.0f)) / (tf + k1_ * norm);
        }
        
        if (score > 0.0f) {
            results.push_back({static_cast<int>(doc_id), score});
        }
    }
    
    // Ordenar por score descendente
    std::sort(results.begin(), results.end(), 
        [](const BM25Result& a, const BM25Result& b) {
            return a.score > b.score;
        });
    
    // Retornar top-k
    if ((int)results.size() > k) {
        results.resize(k);
    }
    
    return results;
}
