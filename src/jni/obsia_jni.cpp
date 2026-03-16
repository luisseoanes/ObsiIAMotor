#include <jni.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include "obsia_api.h"

#ifdef __ANDROID__
#  include <android/log.h>
#  define LOG_TAG "ObsiaJNI"
#  define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#  define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#  define LOGI(...) printf(__VA_ARGS__)
#  define LOGE(...) fprintf(stderr, __VA_ARGS__)
#endif

// Helper to escape JSON strings
std::string jsonEscape(const std::string& s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(c);
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

// Struct to hold JNI streaming callback context
struct StreamingContext {
    JNIEnv* env;
    jobject callback_obj;
    jmethodID on_token_method;
    bool should_stop;
};

// C callback bridge for streaming
static int jni_streaming_callback(const char* token_text, void* user_data) {
    StreamingContext* ctx = static_cast<StreamingContext*>(user_data);
    if (!ctx || ctx->should_stop) return 1;

    jstring jtoken = ctx->env->NewStringUTF(token_text);
    if (!jtoken) return 1;

    // Call the Java/Kotlin callback: onToken(String) -> boolean
    jboolean should_continue = ctx->env->CallBooleanMethod(ctx->callback_obj, ctx->on_token_method, jtoken);
    ctx->env->DeleteLocalRef(jtoken);

    if (ctx->env->ExceptionCheck()) {
        ctx->env->ExceptionClear();
        ctx->should_stop = true;
        return 1;
    }

    return should_continue ? 0 : 1;
}

extern "C" {

JNIEXPORT jint JNICALL
Java_com_obsIA_engine_NativeEngine_init(
    JNIEnv* env, jobject thiz, jstring modelPath, jstring ragPath, jint nThreads) {
    
    if (modelPath == nullptr || ragPath == nullptr) {
        return -1;
    }

    const char* nativeModelPath = env->GetStringUTFChars(modelPath, nullptr);
    const char* nativeRagPath = env->GetStringUTFChars(ragPath, nullptr);

    ObsiaConfig config;
    config.model_path = nativeModelPath;
    config.rag_chunks_path = nativeRagPath;
    config.n_threads = 0;       // 0 = auto-detect (big cores or cores - 1)
    config.n_ctx = 1024;        // Optimized for mobile RAM usage
    config.n_batch = 128;       // Default for mobile
    config.rag_k = 3;           // Default number of chunks

    int result = obsia_init(&config);

    env->ReleaseStringUTFChars(modelPath, nativeModelPath);
    env->ReleaseStringUTFChars(ragPath, nativeRagPath);

    LOGI("Obsia Engine initialized with result: %d\n", result);
    return (jint)result;
}

JNIEXPORT jstring JNICALL
Java_com_obsIA_engine_NativeEngine_processQuery(
    JNIEnv* env, jobject thiz, jstring query) {
    
    auto t_start = std::chrono::steady_clock::now();

    if (query == nullptr) {
        return env->NewStringUTF("{\"status\":\"error\", \"error_message\":\"Query is null\"}");
    }

    const char* nativeQuery = env->GetStringUTFChars(query, nullptr);
    
    // Call the core API
    const char* response = obsia_chat(nativeQuery);
    
    env->ReleaseStringUTFChars(query, nativeQuery);

    auto t_end = std::chrono::steady_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    // Wrap the response in a JSON compatible with the Android App contract
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"ok\","
        << "\"response_text\":\"" << jsonEscape(response ? response : "") << "\","
        << "\"processing_ms\":" << duration_ms
        << "}";

    return env->NewStringUTF(oss.str().c_str());
}

// Streaming version: calls Kotlin callback for each token
JNIEXPORT jstring JNICALL
Java_com_obsIA_engine_NativeEngine_processQueryStreaming(
    JNIEnv* env, jobject thiz, jstring query, jobject tokenCallback) {

    auto t_start = std::chrono::steady_clock::now();

    if (query == nullptr) {
        return env->NewStringUTF("{\"status\":\"error\", \"error_message\":\"Query is null\"}");
    }

    const char* nativeQuery = env->GetStringUTFChars(query, nullptr);
    const char* response = nullptr;

    if (tokenCallback != nullptr) {
        // Get the callback interface method
        jclass callbackClass = env->GetObjectClass(tokenCallback);
        jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)Z");

        if (onTokenMethod != nullptr) {
            StreamingContext ctx;
            ctx.env = env;
            ctx.callback_obj = tokenCallback;
            ctx.on_token_method = onTokenMethod;
            ctx.should_stop = false;

            response = obsia_chat_streaming(nativeQuery, jni_streaming_callback, &ctx);
        } else {
            // Fallback to non-streaming if method not found
            response = obsia_chat(nativeQuery);
        }
    } else {
        response = obsia_chat(nativeQuery);
    }

    env->ReleaseStringUTFChars(query, nativeQuery);

    auto t_end = std::chrono::steady_clock::now();
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"ok\","
        << "\"response_text\":\"" << jsonEscape(response ? response : "") << "\","
        << "\"processing_ms\":" << duration_ms
        << "}";

    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_obsIA_engine_NativeEngine_isReady(
    JNIEnv* env, jobject thiz) {
    return obsia_is_ready() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_obsIA_engine_NativeEngine_release(
    JNIEnv* env, jobject thiz) {
    obsia_free();
    LOGI("Obsia Engine released\n");
}

JNIEXPORT jint JNICALL
Java_com_obsIA_engine_NativeEngine_getMemoryUsageMB(
    JNIEnv* env, jobject thiz) {
#ifdef __ANDROID__
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) return -1;

    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long long kb = 0;
            iss >> kb;
            return static_cast<jint>((kb + 1023LL) / 1024LL);
        }
    }
#endif
    return -1;
}

} // extern "C"
