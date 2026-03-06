cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pcl_sphere_fitting)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(main main.cpp)

target_link_libraries(main ${PCL_LIBRARIES})
