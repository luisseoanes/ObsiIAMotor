#include "bm25.h"
#include "text_norm.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <cctype>
#include <thread>

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
    
    precompute_idf();
}

void BM25Ranker::precompute_idf() {
    idf_cache_.clear();
    float N = (float)doc_term_freqs_.size();
    for (const auto& [term_id, df] : doc_freqs_) {
        idf_cache_[term_id] = std::log((N - (float)df + 0.5f) / ((float)df + 0.5f) + 1.0f);
    }
}

std::vector<BM25Result> BM25Ranker::search(const std::string& query, int k) {
    auto query_tokens = tokenize(query);
    
    std::vector<int> query_term_ids;
    for (const auto& term : query_tokens) {
        int id = get_term_id(term, false);
        if (id != -1) {
            query_term_ids.push_back(id);
        }
    }

    const size_t n_docs = doc_term_freqs_.size();
    if (n_docs == 0 || query_term_ids.empty()) return {};

    const int n_threads = std::min((int)std::thread::hardware_concurrency(), std::min(4, (int)n_docs));
    
    if (n_threads <= 1 || n_docs < 50) {
        std::vector<BM25Result> results;
        results.reserve(n_docs);
        for (size_t doc_id = 0; doc_id < n_docs; ++doc_id) {
            float score = 0.0f;
            for (int term_id : query_term_ids) {
                auto it = doc_term_freqs_[doc_id].find(term_id);
                if (it == doc_term_freqs_[doc_id].end()) continue;
                auto idf_it = idf_cache_.find(term_id);
                float idf = (idf_it != idf_cache_.end()) ? idf_it->second : 0.0f;
                int tf = it->second;
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
        std::sort(results.begin(), results.end(), 
            [](const BM25Result& a, const BM25Result& b) { return a.score > b.score; });
        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    std::vector<std::vector<BM25Result>> thread_results(n_threads);
    std::vector<std::thread> threads;

    auto score_range = [&](int thread_id, size_t start, size_t end) {
        auto& local_results = thread_results[thread_id];
        for (size_t doc_id = start; doc_id < end; ++doc_id) {
            float score = 0.0f;
            for (int term_id : query_term_ids) {
                auto it = doc_term_freqs_[doc_id].find(term_id);
                if (it == doc_term_freqs_[doc_id].end()) continue;
                auto idf_it = idf_cache_.find(term_id);
                float idf = (idf_it != idf_cache_.end()) ? idf_it->second : 0.0f;
                int tf = it->second;
                float doc_len = (float)doc_lengths_[doc_id];
                float norm = 1.0f;
                if (avg_doc_length_ > 0.0f) {
                    norm = 1.0f - b_ + b_ * (doc_len / avg_doc_length_);
                }
                score += idf * (tf * (k1_ + 1.0f)) / (tf + k1_ * norm);
            }
            if (score > 0.0f) {
                local_results.push_back({static_cast<int>(doc_id), score});
            }
        }
    };

    size_t chunk_size = n_docs / n_threads;
    for (int t = 0; t < n_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = (t == n_threads - 1) ? n_docs : start + chunk_size;
        threads.emplace_back(score_range, t, start, end);
    }
    for (auto& th : threads) th.join();

    std::vector<BM25Result> results;
    for (auto& tr : thread_results) {
        results.insert(results.end(), tr.begin(), tr.end());
    }

    std::sort(results.begin(), results.end(),
        [](const BM25Result& a, const BM25Result& b) { return a.score > b.score; });
    if ((int)results.size() > k) results.resize(k);
    return results;
}
