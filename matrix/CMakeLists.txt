cmake_minimum_required(VERSION 3.14)

project(Matrix LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(USE_EIGEN "Use Eigen's matrix implementation" ON)
if (USE_EIGEN)
    message("Matrix using Eigen's implementation")
    add_definitions(-DUSE_EIGEN)
    set_directory_properties(path eigen PROPERTIES
        COMPILE_OPTIONS "-w"
    )
    include_directories(eigen)
endif (USE_EIGEN)


set(SOURCES
    matrix.cpp
)

add_library(
    Matrix
    ${SOURCES}
)

target_include_directories(
    Matrix
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
