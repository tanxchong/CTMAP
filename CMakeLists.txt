project(test)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)


#set(CMAKE_C_COMPILER    "/local/bin/gcc")
#set(CMAKE_CXX_COMPILER  "/local/bin/g++")


find_package(OpenCV REQUIRED)



include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})  #xhjia

#add_executable(test main.cpp)
#target_link_libraries(test ${OpenCV_LIBS})

#set(PCL_DIR "/usr/local/Cellar/opencv/2.4.11_1")
#link_directories()
#message("GFlags include dir: "  ${OpenCV_LIBS})

# up first.
#find_package(GLog REQUIRED)
#include_directories(${GLOG_INCLUDE_DIR})
#set(LIBS ${LIBS} ${GLOG_LIBS})
# for static linking with GLOG, GLOG_STATIC_LIB is set in GLOG's find module
# add_library(glogstatic STATIC IMPORTED)
# TODO: Is this directive required for all libraries? Seems to make no difference.
#set_target_properties(glogstatic PROPERTIES IMPORTED_LOCATION ${GLOG_STATIC_LIB})
#message(STATUS "GLog include dir: " ${GLOG_INCLUDE_DIR} ${LIBS})
# find GFlags headers and libs (needed for GLog)
#find_package(GFlags REQUIRED)
#include_directories(${GFLAGS_INCLUDE_DIR})
#set(LIBS ${LIBS} ${GFLAGS_LIBS})
# for static linking with GFLAGS, GFLAGS_STATIC_LIB is set in GFLAGS' find module
#add_library(gflagsstatic STATIC IMPORTED)
#set_target_properties(gflagsstatic PROPERTIES IMPORTED_LOCATION ${GFLAGS_STATIC_LIB})
#message(STATUS "GFlags include dir: " ${GFLAGS_INCLUDE_DIR} ${LIBS})



add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS})
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

