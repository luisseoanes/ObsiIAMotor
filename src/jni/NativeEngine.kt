package com.obsIA.engine

/**
 * Android Native Bridge for the Obsia Medical Engine in Kotlin.
 * Place this file in: app/src/main/java/com/obsIA/engine/NativeEngine.kt
 * (Ensure the package name matches your Android project structure)
 */
class NativeEngine {

    companion object {
        init {
            // This loads 'libobsia_jni.so'
            System.loadLibrary("obsia_jni")
        }
    }

    /**
     * Callback interface for streaming tokens.
     * Implement this to receive tokens as they are generated.
     */
    interface TokenCallback {
        /**
         * Called for each generated token piece.
         * @param token The text piece just generated.
         * @return true to continue generation, false to stop early.
         */
        fun onToken(token: String): Boolean
    }

    /**
     * Initializes the engine.
     * @param modelPath Path to the .gguf model file.
     * @param ragPath Path to the chunks.json file.
     * @param nThreads Number of threads (0 = auto-detect optimal count).
     * @return 0 on success, negative values on error.
     */
    external fun init(modelPath: String, ragPath: String, nThreads: Int): Int

    /**
     * Sends a message to the medical assistant (blocking).
     * @param query User medical symptoms or question.
     * @return JSON string with "status", "response_text", and "processing_ms".
     */
    external fun processQuery(query: String): String

    /**
     * Sends a message with streaming (tokens arrive in real-time via callback).
     * @param query User medical symptoms or question.
     * @param callback Receives each token as it's generated.
     * @return JSON string with final "status", "response_text", and "processing_ms".
     */
    external fun processQueryStreaming(query: String, callback: TokenCallback?): String

    /**
     * Checks if the engine is ready for queries.
     */
    external fun isReady(): Boolean

    /**
     * Releases memory and model resources.
     */
    external fun release()

    /**
     * Returns the current memory usage of the process in MB.
     */
    external fun getMemoryUsageMB(): Int
}
