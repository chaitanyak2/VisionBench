find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB TEST_SOURCES core/*.cpp plugins/*.cpp)
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests GTest::GTest GTest::Main pthread visionbench_core)

add_test(NAME AllTests COMMAND run_tests)
