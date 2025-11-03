cmake_minimum_required(VERSION 3.16)
project(visionbench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# ----------------------------
# LibTorch
# ----------------------------
set(TORCH_DIR "/opt/libtorch" CACHE PATH "Path to LibTorch installation")
set(CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)

# ----------------------------
# GStreamer
# ----------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0)

include_directories(${GST_INCLUDE_DIRS})
link_directories(${GST_LIBRARY_DIRS})

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


#-----------------------------
# FASISS
#-----------------------------
# Enable OpenMP for all targets that need it
find_package(OpenMP REQUIRED)
find_library(BLAS_LIB openblas)
find_library(LAPACK_LIB lapack)
add_library(Faiss::faiss SHARED IMPORTED)
set_target_properties(Faiss::faiss PROPERTIES
    IMPORTED_LOCATION /usr/local/lib/libfaiss.so
    INTERFACE_INCLUDE_DIRECTORIES /usr/local/include
)
# ----------------------------
# Subdirectories
# ----------------------------
add_subdirectory(core)
add_subdirectory(plugins)
enable_testing()
add_subdirectory(tests)

# RPATH so plugin finds libtorch at runtime
#set_target_properties(gstpreproc PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
#set_target_properties(gsttorchinfer PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
# Install targets optionally
