cmake_minimum_required(VERSION 3.0)
project(openmp_hello_world)

set(CMAKE_CXX_STANDARD 17)

add_executable(openmp_hello_world main.cpp)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(openmp_hello_world PUBLIC OpenMP::OpenMP_CXX)
endif ()