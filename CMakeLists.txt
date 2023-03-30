cmake_minimum_required(VERSION 3.16)
project(ParallelMatrixMul)
set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(SOURCES
        MatrixUtils.cpp
        ParallelMatrixMultiplier.cpp)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_executable(ParallelMatrixMul ${SOURCES})
