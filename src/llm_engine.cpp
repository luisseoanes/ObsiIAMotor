#include "llm_engine.h"
#include <cstdio>
#include <algorithm>
#include <iostream>

LLMEngine::LLMEngine() {}

LLMEngine::~LLMEngine() {
    unload_model();
}

void LLMEngine::unload_model() {
    if (batch_inited_) { llama_batch_free(batch); batch_inited_ = false; }
    if (smpl) { llama_sampler_free(smpl); smpl = nullptr; }
    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
}

bool LLMEngine::load_model(const std::string& model_path, int n_ctx, int n_batch, int n_threads) {
    unload_model();
    n_batch_ = n_batch;
    system_tokens_len_ = 0;

    auto log_cb = [](enum ggml_log_level level, const char* text, void* /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    };
    llama_log_set(log_cb, nullptr);

    llama_model_params model_params = llama_model_default_params();
#ifdef GGML_USE_VULKAN
    model_params.n_gpu_layers = 99; // Usar GPU (Vulkan) en movil si esta disponible
#else
    model_params.n_gpu_layers = 0;  // Fallback a CPU
#endif
    model_params.use_mmap = true;  // Fuerza la lectura desde disco para ahorrar memoria RAM activa (CRITICO para movil)
    model_params.use_mlock = false; // Permite Android paginar la memoria si el OS necesita espacio

    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error loading model: %s\n", model_path.c_str());
        return false;
    }

    vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = std::min(n_batch, 64);
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = std::min(n_threads, 4);
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.type_k = GGML_TYPE_Q4_0;
    ctx_params.type_v = GGML_TYPE_Q4_0;

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error creating context\n");
        unload_model();
        return false;
    }

    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!smpl) {
        fprintf(stderr, "Error creating sampler\n");
        unload_model();
        return false;
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.50f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    batch = llama_batch_init(n_batch, 0, 1);
    batch_inited_ = true;
    return true;
}

void LLMEngine::ensure_kv_space(int n_needed, int n_keep_system) {
    if (!ctx) return;
    const int n_ctx = llama_n_ctx(ctx);
    int n_past = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    if (n_past < 0) n_past = 0;

    if (n_past + n_needed > n_ctx) {
        int n_discard = (n_past - n_keep_system) / 2;
        if (n_discard < n_needed) n_discard = n_needed;

        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_keep_system, n_keep_system + n_discard);
        llama_memory_seq_add(llama_get_memory(ctx), 0, n_keep_system + n_discard, n_past, -n_discard);
    }
}

void LLMEngine::reset_context() {
    if (!ctx) return;
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
    system_tokens_len_ = 0;
}

bool LLMEngine::run_inference(const std::vector<llama_token>& tokens, bool is_system) {
    if (is_system) {
        if ((int)tokens.size() > llama_n_ctx(ctx)) {
            return false;
        }
    } else {
        ensure_kv_space((int)tokens.size() + 100, system_tokens_len_);
    }

    int n_past = 0;
    int max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (max_pos != -1) n_past = max_pos + 1;

    for (int i = 0; i < (int)tokens.size(); i += n_batch_) {
        int n_eval = std::min((int)tokens.size() - i, n_batch_);

        batch.n_tokens = n_eval;
        for (int k = 0; k < n_eval; ++k) {
            batch.token[k] = tokens[i + k];
            batch.pos[k] = n_past + i + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = false;
        }

        if (i + n_eval == (int)tokens.size()) {
            batch.logits[n_eval - 1] = true;
        }

        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
    }
    return true;
}

bool LLMEngine::set_system_prompt(const std::string& formatted_system_prompt) {
    if (!ctx || !vocab) return false;

    // Reset context
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);
    
    std::vector<llama_token> sys_tokens(formatted_system_prompt.size() + 2);
    int n_sys = llama_tokenize(vocab, formatted_system_prompt.c_str(), formatted_system_prompt.size(), sys_tokens.data(), sys_tokens.size(), true, true);
    if (n_sys > 0) {
        sys_tokens.resize(n_sys);
        if (run_inference(sys_tokens, true)) {
            system_tokens_len_ = n_sys;
            return true;
        }
    }
    return false;
}

std::string LLMEngine::generate_response(const std::string& formatted_user_prompt) {
    if (!ctx || !vocab) return "";
    if (!smpl) return "";
    if (smpl) llama_sampler_reset(smpl);
    if (system_tokens_len_ == 0) {
        reset_context();
    }

    std::vector<llama_token> tokens(formatted_user_prompt.size() + 2);
    int n = llama_tokenize(vocab, formatted_user_prompt.c_str(), formatted_user_prompt.size(), tokens.data(), tokens.size(), false, true);
    if (n <= 0) return "";
    tokens.resize(n);

    if (!run_inference(tokens, false)) return "";
    if (smpl) {
        for (const auto& t : tokens) {
            llama_sampler_accept(smpl, t);
        }
    }

    std::string response;
    int n_past = 0;
    int max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (max_pos != -1) n_past = max_pos + 1;

    llama_token new_token_id;
    int gen_count = 0;
    const int MAX_GEN_TOKENS = 512;
    std::vector<llama_token> recent_tokens;

    while (true) {
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) break;
        if (smpl) llama_sampler_accept(smpl, new_token_id);

        gen_count++;
        if (gen_count >= MAX_GEN_TOKENS) break;

        recent_tokens.push_back(new_token_id);
        if (recent_tokens.size() >= 12) {
            size_t sz = recent_tokens.size();
            bool repeating = true;
            for (int r = 1; r < 4 && repeating; r++) {
                for (int p = 0; p < 3 && repeating; p++) {
                    if (recent_tokens[sz - 12 + p] != recent_tokens[sz - 12 + r*3 + p]) {
                        repeating = false;
                    }
                }
            }
            if (repeating) break;
        }

        char buf[256];
        int n_len = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n_len < 0) break;
        response += std::string(buf, (size_t)n_len);

        if (n_past + 1 > llama_n_ctx(ctx)) {
            ensure_kv_space(10, system_tokens_len_);
            max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            if (max_pos != -1) n_past = max_pos + 1;
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token_id;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        n_past++;

        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }

    // Trim trailing memory to clear context for the next turn
    max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (max_pos + 1 > system_tokens_len_) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, system_tokens_len_, max_pos + 1);
    }

    // Cleanup leading spaces
    size_t start = 0;
    while (start < response.size() && std::isspace(static_cast<unsigned char>(response[start]))) {
        start++;
    }
    if (start > 0) response = response.substr(start);

    return response;
}
