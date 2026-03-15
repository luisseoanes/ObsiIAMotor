#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct BM25Result {
    int chunk_id;
    float score;
};

class BM25Ranker {
public:
    BM25Ranker(float k1 = 1.5f, float b = 0.75f);
    
    // Indexar chunks
    void index_documents(const std::vector<std::string>& docs);
    
    // Buscar top-k
    std::vector<BM25Result> search(const std::string& query, int k = 10);
    
private:
    float k1_, b_;
    
    // Optimizacion RAM: Usar IDs enteros en lugar de strings
    std::unordered_map<std::string, int> vocab_;
    int next_term_id_ = 0;
    
    // Mapa: doc_id -> (term_id -> count)
    std::vector<std::unordered_map<int, int>> doc_term_freqs_;
    
    // Mapa: term_id -> doc_count (en cuantos docs aparece el termino)
    std::unordered_map<int, int> doc_freqs_;
    
    std::vector<int> doc_lengths_;
    float avg_doc_length_;
    
    std::vector<std::string> tokenize(const std::string& text);
    int get_term_id(const std::string& term, bool create_if_missing);
    float compute_idf(int term_id);
};
