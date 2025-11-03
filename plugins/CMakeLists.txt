set(PLUGIN_COMMON_LIBS ${GST_LIBRARIES} ${TORCH_LIBRARIES} pthread)

# vbpreprocessor
add_library(vbpreprocessor MODULE vbprocessor/torch_std_mean_preproc.cpp)
target_include_directories(vbpreprocessor PRIVATE ${GST_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(vbpreprocessor ${PLUGIN_COMMON_LIBS})
set_target_properties(vbpreprocessor PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
target_link_libraries(vbpreprocessor coreutils)

# vbinfer
add_library(vbinfer MODULE vbinfer/torch_infer.cpp)
target_include_directories(vbinfer PRIVATE ${GST_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(vbinfer ${PLUGIN_COMMON_LIBS})
set_target_properties(vbinfer PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
target_link_libraries(vbinfer coreutils)


message(STATUS "Plugins built: vbpreprocessor, vbinfer")