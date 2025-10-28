cmake_minimum_required(VERSION 3.18)
project(VisionBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_PLUGINS "Build GStreamer plugins" ON)

# Dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_BASE REQUIRED gstreamer-base-1.0)
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)

include_directories(${GSTREAMER_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${SQLite3_INCLUDE_DIRS})

add_subdirectory(core)
add_subdirectory(plugins)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
