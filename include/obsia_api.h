#ifndef OBSIA_API_H
#define OBSIA_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialization parameters
typedef struct {
    const char* model_path;      // Path to the .gguf file
    const char* rag_chunks_path; // Path to the chunks.json file
    int n_threads;               // Number of CPU threads (0 = auto-detect cores-1)
    int n_ctx;                   // Context window (default 1024)
    int n_batch;                 // Batch size (default 128)
    int rag_k;                   // Number of RAG chunks to inject (default 3)
} ObsiaConfig;

// Callback for streaming tokens. Return 0 to continue, non-zero to stop.
typedef int (*obsia_token_callback)(const char* token_text, void* user_data);

// Initialize the engine. Returns 0 on success, non-zero on failure.
int obsia_init(const ObsiaConfig* config);

// Release all resources and unload the model.
void obsia_free();

// Returns 1 if the engine is initialized, 0 otherwise.
int obsia_is_ready();

// Chat directly with the model (Applies RAG + Clinical rules implicitly)
// Returns a pointer to an internal buffer containing the response.
// Do not free the returned pointer. It remains valid until the next call to obsia_chat.
const char* obsia_chat(const char* user_message);

// Streaming version: calls callback for each generated token piece.
// Returns the full response text (same as obsia_chat).
const char* obsia_chat_streaming(const char* user_message, obsia_token_callback callback, void* user_data);

#ifdef __cplusplus
}
#endif

#endif // OBSIA_API_H
