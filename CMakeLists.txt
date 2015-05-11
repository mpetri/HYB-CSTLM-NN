cmake_minimum_required(VERSION 2.8.7)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)

project(knm CXX C)
set(CMAKE_BUILD_TYPE "Release")

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
    append_cxx_compiler_flags("-std=c++11 -Wall -Wextra  -DNDEBUG" "GCC" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-O3 -ffast-math -funroll-loops" "GCC" CMAKE_CXX_FLAGS)
else()
    append_cxx_compiler_flags("-std=c++11 -DNDEBUG" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-stdlib=libc++" "CLANG" CMAKE_CXX_FLAGS)
    append_cxx_compiler_flags("-O3 -ffast-math -funroll-loops" "CLANG" CMAKE_CXX_FLAGS)
endif()
include(CheckAVX2)
if( BUILTIN_INTERLEAVE )
    if( CMAKE_COMPILER_IS_GNUCXX )
        append_cxx_compiler_flags("-msse4.2 -mavx2 -mbmi2" "GCC" CMAKE_CXX_FLAGS)
    else()
        append_cxx_compiler_flags("-msse4.2 -mavx2 -mbmi2" "CLANG" CMAKE_CXX_FLAGS)
    endif()
    message(STATUS "CPU does support AVX/BMI2.")
    add_definitions(-DUSE_BUILTIN_INTERLEAVE)
else()
	message(STATUS "CPU does NOT support AVX/BMI2")
endif()

add_subdirectory(external/sdsl-lite)
add_subdirectory(external/googletest)

include_directories(${CMAKE_HOME_DIRECTORY}/include
                    ${CMAKE_HOME_DIRECTORY}/external/googletest/include
                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/include
                    ${CMAKE_BINARY_DIR}/external/sdsl-lite/external/libdivsufsort-2.0.1/include/
                    ${CMAKE_HOME_DIRECTORY}/external/easyloggingpp/src/
                    )

add_executable(build-index.x src/build-index.cpp)
target_link_libraries(build-index.x sdsl pthread divsufsort divsufsort64)

add_executable(create-collection.x src/create-collection.cpp)
target_link_libraries(create-collection.x sdsl pthread divsufsort divsufsort64)

add_executable(query-index-knm.x src/query-index-knm.cpp)
target_link_libraries(query-index-knm.x sdsl pthread divsufsort divsufsort64)

add_executable(query-index-stupid.x src/query-index-stupid.cpp)
target_link_libraries(query-index-stupid.x sdsl pthread divsufsort divsufsort64)

add_executable(unit-test.x src/unit-test.cpp)
target_link_libraries(unit-test.x gtest_main sdsl pthread divsufsort divsufsort64)
target_compile_options(unit-test.x PUBLIC -DLMSDSL_NO_LOGGING)
enable_testing()
add_test(TestsPass unit-test.x)