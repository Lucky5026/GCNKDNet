cmake_minimum_required(VERSION 2.8.3)
project(preprocess)

add_compile_options(-std=c++11)
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(PCL REQUIRED)
include_directories(
    ${PCL_INCLUDE_DIRS}
)

add_executable(preprocess src/preprocess.cpp)
target_link_libraries(preprocess ${PCL_LIBRARIES})