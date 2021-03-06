cmake_minimum_required(VERSION 2.8.7)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)

project(cstlm CXX C)
set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

option (TIMINGS "Enable more fine grained timings." OFF)
option (CUDA_DYNET "Enable cuda support for dynet." OFF)

if(CUDA_DYNET)
    set(BACKEND "cuda")
    append_cxx_compiler_flags("-DHAVE_CUDA" "GCC" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-DHAVE_CUDA" "CLANG" CMAKE_CXX_FLAGS)
else()
    set(BACKEND "eigen")
endif()

macro(remove_cxx_flag flag)
  string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endmacro()

# C++11 compiler Check
if(NOT CMAKE_CXX_COMPILER_VERSION) # work around for cmake versions smaller than 2.8.10
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
endif()
if(CMAKE_CXX_COMPILER MATCHES ".*clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()
if( (CMAKE_COMPILER_IS_GNUCXX AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 4.7) OR
    (CMAKE_COMPILER_IS_CLANGXX AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.2))
  message(FATAL_ERROR "Your C++ compiler does not support C++11. Please install g++ 4.7 (or greater) or clang 3.2 (or greater)")
else()
  message(STATUS "Compiler is recent enough to support C++11.")
endif()
if( CMAKE_COMPILER_IS_GNUCXX )
    append_cxx_compiler_flags("-g -std=c++11 -Wall -Wextra " "GCC" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-ffast-math -funroll-loops" "GCC" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-fcilkplus -fopenmp -O2" "GCC" CMAKE_CXX_FLAGS)
    remove_cxx_flag("-O3")
else()
    append_cxx_compiler_flags("-g -std=c++11 -Wall" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-stdlib=libc++" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-ffast-math -funroll-loops" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-fcilkplus" "CLANG" CMAKE_CXX_FLAGS)
endif()
include(CheckAVX2)
if( BUILTIN_POPCNT )
    if( CMAKE_COMPILER_IS_GNUCXX )
        append_cxx_compiler_flags("-msse4.2 -DHAVE_SSE42" "GCC" CMAKE_CXX_FLAGS)
    else()
        append_cxx_compiler_flags("-msse4.2 -DHAVE_SSE42" "CLANG" CMAKE_CXX_FLAGS)
    endif()
    message(STATUS "CPU does support fast popcnt.")
else()
	message(STATUS "CPU does NOT support fast popcnt")
endif()

remove_cxx_flag("-O3")
if( TIMINGS )
    if( CMAKE_COMPILER_IS_GNUCXX )
        append_cxx_compiler_flags("-DENABLE_CSTLM_TIMINGS" "GCC" CMAKE_CXX_FLAGS)
    else()
        append_cxx_compiler_flags("-DENABLE_CSTLM_TIMINGS" "CLANG" CMAKE_CXX_FLAGS)
    endif()
endif()

add_subdirectory(external/parallel-range-lite)

add_subdirectory(external/sdsl-lite)

set(EIGEN3_INCLUDE_DIR ${CMAKE_HOME_DIRECTORY}/external/eigen)
add_subdirectory(external/dynet)

include_directories(${CMAKE_HOME_DIRECTORY}/include/
                    ${CMAKE_HOME_DIRECTORY}/include/cstlm/
                    ${CMAKE_HOME_DIRECTORY}/include/nn/
                    ${CMAKE_HOME_DIRECTORY}/external/dynet
                    ${CMAKE_HOME_DIRECTORY}/external/eigen
                    ${CMAKE_HOME_DIRECTORY}/external/googletest/include
                    ${CMAKE_HOME_DIRECTORY}/external/mem_monitor/
                    ${CMAKE_BINARY_DIR}/external/parallel-range-lite/include
                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/include
                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/external/libdivsufsort/include/
                    )

if(CUDA_DYNET)
	link_directories("/usr/local/cuda/lib64/")
	include_directories("/usr/local/cuda/include/")
endif()


add_executable(build-index.x src/build-index.cpp)
target_link_libraries(build-index.x sdsl pthread divsufsort divsufsort64 cilkrts)

add_executable(create-collection.x src/create-collection.cpp)
target_link_libraries(create-collection.x sdsl pthread divsufsort divsufsort64 cilkrts prange)

add_executable(query-index-knm.x src/query-index-knm.cpp)
target_link_libraries(query-index-knm.x sdsl pthread divsufsort divsufsort64 cilkrts)

add_executable(unit-test.x src/unit-test.cpp)
target_link_libraries(unit-test.x gtest_main sdsl pthread divsufsort divsufsort64 cilkrts)
enable_testing()
add_test(TestsPass unit-test.x)


### NEW HYBRID STUFF
#add_executable(train-hyblm.x src/train-hyblm.cpp)
#target_link_libraries(train-hyblm.x sdsl pthread divsufsort divsufsort64 cilkrts dynet)

remove_cxx_flag("-O3") # fix for parallel integer SA construction
add_executable(build-and-bench-cstlm.x src/build-and-bench-cstlm.cpp)

add_executable(cstlm-eval-ppl-tau.x src/cstlm-eval-ppl-tau.cpp)

add_executable(build-and-bench-rnnlm.x src/build-and-bench-rnnlm.cpp)

add_executable(build-hyblm.x src/build-hyblm.cpp)
add_executable(bench-hyblm.x src/bench-hyblm.cpp)

add_executable(precompute-ngrams.x src/precompute-ngrams.cpp)
target_link_libraries(precompute-ngrams.x prange sdsl pthread divsufsort divsufsort64 cilkrts)

if(CUDA_DYNET)
target_link_libraries(build-and-bench-rnnlm.x sdsl prange pthread divsufsort divsufsort64 cilkrts gdynet dynetcuda cublas)
target_link_libraries(build-and-bench-cstlm.x sdsl prange pthread divsufsort divsufsort64 cilkrts gdynet dynetcuda cublas)
target_link_libraries(cstlm-eval-ppl-tau.x sdsl prange pthread divsufsort divsufsort64 cilkrts gdynet dynetcuda cublas)
target_link_libraries(build-hyblm.x sdsl prange pthread divsufsort divsufsort64 cilkrts gdynet dynetcuda cublas)
target_link_libraries(bench-hyblm.x sdsl prange pthread divsufsort divsufsort64 cilkrts gdynet dynetcuda cublas)
else()
target_link_libraries(build-and-bench-cstlm.x prange sdsl pthread divsufsort divsufsort64 cilkrts dynet)
target_link_libraries(cstlm-eval-ppl-tau.x prange sdsl pthread divsufsort divsufsort64 cilkrts dynet)
target_link_libraries(build-and-bench-rnnlm.x prange sdsl pthread divsufsort divsufsort64 cilkrts dynet)
target_link_libraries(build-hyblm.x prange sdsl pthread divsufsort divsufsort64 cilkrts dynet)
target_link_libraries(bench-hyblm.x prange sdsl pthread divsufsort divsufsort64 cilkrts dynet)
endif()
