find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

add_library(coreutils STATIC gstcoremeta.cpp benchmark_recorder.cpp)
target_include_directories(coreutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(coreutils PUBLIC ${SQLITE3_INCLUDE_DIRS})
target_link_libraries(coreutils PUBLIC ${SQLITE3_LIBRARIES} ${GST_LIBRARIES})

