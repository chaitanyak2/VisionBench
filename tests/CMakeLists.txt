enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Define helper
function(add_gtest name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name}
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        pthread
        ${TORCH_LIBRARIES}
        ${GST_LIBRARIES}
        ${GST_VIDEO_LIBRARIES}
        coreutils
        gtest
        gtest_main
        gstfaissstore
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# --- Tests that use plugin sources ---
add_gtest(test_torchpreproc
${CMAKE_SOURCE_DIR}/plugins/vbprocessor/torch_std_mean_preproc.cpp
    unit/test_torch_std_mean_preproc.cpp
)

add_gtest(test_torchinfer
${CMAKE_SOURCE_DIR}/plugins/vbinfer/torch_infer.cpp
    unit/test_torch_infer.cpp
)

# --- Other core tests ---
add_gtest(test_benchmark_recorder test_benchmark_recorder.cpp)
add_gtest(test_gstcoremeta core/test_gstcoremeta.cpp)
add_gtest(test_metadata core/test_metadata.cpp)
add_gtest(test_faiss_storage  unit/test_faiss_storage.cpp)
add_gtest(test_faissstore_sink  unit/test_faissstore_sink.cpp)
