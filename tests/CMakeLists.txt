# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

find_package(GTest REQUIRED)

add_executable(test_benchmark test_benchmark_recorder.cpp)
target_include_directories(test_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
target_link_libraries(test_benchmark PRIVATE visionbench GTest::GTest GTest::Main pthread sqlite3)

enable_testing()
add_test(NAME BenchTests COMMAND test_benchmark)
