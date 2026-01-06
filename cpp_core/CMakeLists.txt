# ============================================================================
# ATL4S-FOREX AGI Ultra C++ Build System
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(Atl4sAGI VERSION 2.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Windows specific
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(WARNING "OpenMP not found - parallel processing disabled")
endif()

# Output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# CORE LIBRARIES
# ============================================================================

# MCTS Ultra
add_library(mcts SHARED
    mcts.cpp
)
target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mcts OpenMP::OpenMP_CXX)
endif()

# Physics Ultra
add_library(physics SHARED
    physics.cpp
)
target_include_directories(physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(physics OpenMP::OpenMP_CXX)
endif()

# HDC Ultra
add_library(hdc SHARED
    hdc.cpp
)
target_include_directories(hdc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(hdc OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# NEW AGI LIBRARIES
# ============================================================================

# Reasoning Core
add_library(reasoning SHARED
    reasoning.cpp
)
target_include_directories(reasoning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(reasoning OpenMP::OpenMP_CXX)
endif()

# Memory Core
add_library(memory SHARED
    memory.cpp
)
target_include_directories(memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(memory OpenMP::OpenMP_CXX)
endif()

# Learning Core
add_library(learning SHARED
    learning.cpp
)
target_include_directories(learning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(learning OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# COMBINED AGI LIBRARY
# ============================================================================

add_library(atl4s_agi SHARED
    mcts.cpp
    physics.cpp
    hdc.cpp
    reasoning.cpp
    memory.cpp
    learning.cpp
)
target_include_directories(atl4s_agi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(atl4s_agi OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

install(TARGETS mcts physics hdc reasoning memory learning atl4s_agi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES 
    mcts.h
    physics.h
    hdc.h
    reasoning.h
    memory.h
    learning.h
    DESTINATION include
)

# ============================================================================
# TESTS (Optional)
# ============================================================================

option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    # MCTS Test
    add_executable(test_mcts tests/test_mcts.cpp)
    target_link_libraries(test_mcts mcts)
    add_test(NAME MCTSTest COMMAND test_mcts)
    
    # Physics Test
    add_executable(test_physics tests/test_physics.cpp)
    target_link_libraries(test_physics physics)
    add_test(NAME PhysicsTest COMMAND test_physics)
    
    # HDC Test
    add_executable(test_hdc tests/test_hdc.cpp)
    target_link_libraries(test_hdc hdc)
    add_test(NAME HDCTest COMMAND test_hdc)
endif()

# ============================================================================
# BENCHMARK (Optional)
# ============================================================================

option(BUILD_BENCHMARKS "Build benchmark executables" OFF)

if(BUILD_BENCHMARKS)
    # Benchmark executable
    add_executable(benchmark_agi benchmarks/benchmark.cpp)
    target_link_libraries(benchmark_agi atl4s_agi)
endif()

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "=== ATL4S AGI Ultra Build Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "")
message(STATUS "Libraries to build:")
message(STATUS "  - mcts (MCTS Ultra)")
message(STATUS "  - physics (Physics Ultra)")
message(STATUS "  - hdc (HDC Ultra)")
message(STATUS "  - reasoning (Reasoning Core)")
message(STATUS "  - memory (Memory Core)")
message(STATUS "  - learning (Learning Core)")
message(STATUS "  - atl4s_agi (Combined)")
message(STATUS "")
