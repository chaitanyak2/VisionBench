set(PLUGIN_COMMON_LIBS ${GST_LIBRARIES} ${TORCH_LIBRARIES} pthread)

# vbpreprocessor
add_library(vbpreprocessor SHARED vbprocessor/torch_std_mean_preproc.cpp)
target_include_directories(vbpreprocessor PRIVATE ${GST_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(vbpreprocessor ${PLUGIN_COMMON_LIBS})
set_target_properties(vbpreprocessor PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
target_link_libraries(vbpreprocessor coreutils)

# vbinfer
add_library(vbinfer SHARED vbinfer/torch_infer.cpp)
target_include_directories(vbinfer PRIVATE ${GST_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(vbinfer ${PLUGIN_COMMON_LIBS})
set_target_properties(vbinfer PROPERTIES BUILD_RPATH "${TORCH_DIR}/lib")
target_link_libraries(vbinfer coreutils)

add_library(gstfaissstore SHARED vbstorage/gstfaissstore.cpp)

target_include_directories(gstfaissstore PRIVATE ${GST_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}) # adjust as needed

target_link_libraries(gstfaissstore coreutils)

# Install plugin into GStreamer plugin dir if you want
# install(TARGETS gstfaissstore DESTINATION ${CMAKE_INSTALL_LIBDIR}/gstreamer-1.0)


message(STATUS "Plugins built: vbpreprocessor, vbinfer")