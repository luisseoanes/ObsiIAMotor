#pragma once
#include "bm25.h"
#include <string>
#include <vector>

enum class ObstetricState;

struct Chunk {
    std::string text;
    std::string source;
    int chunk_id = 0;
    int word_count = 0;
    std::string section_type;
    std::string clinical_topic;
};

class RAGEngine {
public:
    // chunks_path: JSON de chunks. Opcionalmente busca embeddings.bin en el mismo directorio para RAG híbrido (BM25 + similitud).
    RAGEngine(const std::string& chunks_path);
    
    std::vector<std::string> search(const std::string& query, int k = 5);
    std::string get_augmented_context(const std::string& query, int k, ObstetricState state);
    
private:
    std::vector<Chunk> chunks_;
    BM25Ranker ranker_;
    std::vector<float> embeddings_;  // row-major: n_chunks * dim
    int embed_dim_ = 0;
    bool has_embeddings_ = false;
    
    void load_chunks(const std::string& path);
    void load_embeddings_if_present(const std::string& chunks_path);
    std::vector<int> top_k_similar_to(int chunk_id, int k) const;
    std::string expand_query_for_obstetrics(const std::string& query) const;
    std::string detect_topic_hint(const std::string& query) const;
    bool is_chunk_compatible(const Chunk& chunk, ObstetricState state) const;
};
