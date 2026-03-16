# Install script for directory: C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/obsia_mobile_lib")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/android/sdk/ndk/29.0.14206865/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-objdump.exe")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/src/libggml.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-cpu.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-alloc.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-backend.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-blas.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-cann.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-cpp.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-cuda.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-opt.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-metal.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-rpc.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-virtgpu.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-sycl.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-vulkan.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-webgpu.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/ggml-zendnn.h"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/llama.cpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/src/libggml-base.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/ggml-config.cmake"
    "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/ggml-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/build_x86_64/llama_build/ggml/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
