cmake_minimum_required(VERSION 3.14)
project(obsia_mobile_lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/llama.cpp")

if (NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "No se encontro llama.cpp en ${LLAMA_DIR}")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

# --- Mobile Optimizations (Android NDK / GPU) ---
if (ANDROID)
    message(STATUS "[Obsia] Compilando para Android - Habilitando Aceleracion GPU (Vulkan)")
    set(LLAMA_VULKAN ON CACHE BOOL "" FORCE)
    # Disable OpenMP to avoid dependency on libomp.so which is not present on many devices
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()


add_subdirectory("${LLAMA_DIR}" llama_build)

set(SOURCES
    src/obsia_api.cpp
    src/llm_engine.cpp
    src/rag_engine.cpp
    src/clinical_engine.cpp
    src/bm25.cpp
)

# For testing locally on Windows easily without export macro headaches, use STATIC.
# When compiling for Android via NDK, build scripts usually override this to SHARED.
add_library(obsia STATIC ${SOURCES})

target_include_directories(obsia
    PUBLIC include
    PRIVATE src
    PRIVATE "${LLAMA_DIR}/include"
    PRIVATE "${LLAMA_DIR}/ggml/include"
    PRIVATE "${LLAMA_DIR}/common"
)

target_link_libraries(obsia PRIVATE llama ggml)

# --- JNI Support (Conditional) ---
if (ANDROID)
    # The NDK automatically provides jni.h
    set(HAS_JNI TRUE)
else()
    find_package(JNI)
    if (JNI_FOUND)
        set(HAS_JNI TRUE)
    endif()
endif()

if (HAS_JNI)
    message(STATUS "[Obsia] JNI Encontrado - Configurando libreria para Android/Java")
    add_library(obsia_jni SHARED src/jni/obsia_jni.cpp)
    target_link_libraries(obsia_jni PRIVATE obsia)
    
    if (ANDROID)
        target_link_libraries(obsia_jni PRIVATE log)
    else()
        target_include_directories(obsia_jni PRIVATE ${JNI_INCLUDE_DIRS})
    endif()
endif()

# Add a test runner executable to verify the library works correctly
add_executable(test_runner src/test_runner.cpp)
target_link_libraries(test_runner PRIVATE obsia)
