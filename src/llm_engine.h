#pragma once

#include <string>
#include <vector>
#include "llama.h"

// Encapsulates llama.cpp logic for pure inference and rolling context
class LLMEngine {
public:
    LLMEngine();
    ~LLMEngine();

    bool load_model(const std::string& model_path, int n_ctx, int n_batch, int n_threads);
    void unload_model();

    // Ingests system prompt. Resets the context if needed.
    bool set_system_prompt(const std::string& system_prompt);

    // Predicts the next response given a complete constructed prompt
    std::string generate_response(const std::string& formatted_user_prompt);

    // Get the underlying llama model & vocab for template applying
    llama_model* get_model() const { return model; }
    const llama_vocab* get_vocab() const { return vocab; }
    llama_context* get_context() const { return ctx; }

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* smpl = nullptr;
    llama_batch batch;
    bool batch_inited_ = false;
    
    int n_batch_ = 128;
    int system_tokens_len_ = 0;

    bool run_inference(const std::vector<llama_token>& tokens, bool is_system);
    void ensure_kv_space(int n_needed, int n_keep_system);
    void reset_context();
};
