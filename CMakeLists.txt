cmake_minimum_required(VERSION 3.0)
project(Parallel)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#set(CMAKE_CXX_FLAGS "-O3")

add_executable(Parallel main.cpp threads.cpp)
