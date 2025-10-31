cmake_minimum_required(VERSION 3.16)
project(visionbench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add library
add_library(visionbench
    core/benchmark_recorder.cpp
)

target_include_directories(visionbench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
find_package(Threads REQUIRED)
target_link_libraries(visionbench PUBLIC Threads::Threads sqlite3)

# tests
enable_testing()
add_subdirectory(tests)

# Install targets optionally
