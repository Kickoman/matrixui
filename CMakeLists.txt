cmake_minimum_required(VERSION 3.16...3.28)

project(MatrixGui LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_GUI "Build GUI version" ON)
option(BUILD_CLI "Build CLI version" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_SHARED_LIBS "Build the matrixgui libraries as shared objects" OFF)

set(MATRIXGUI_LIB_TYPE STATIC)
if(BUILD_SHARED_LIBS)
    set(MATRIXGUI_LIB_TYPE SHARED)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

add_library(matrixgui_base INTERFACE)
target_include_directories(matrixgui_base INTERFACE ${PROJECT_SOURCE_DIR})
target_compile_features(matrixgui_base INTERFACE cxx_std_20)
target_compile_options(matrixgui_base INTERFACE -Wall -Wextra)

add_library(matrixgui_contrib INTERFACE)
target_include_directories(matrixgui_contrib SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/contrib)

add_library(matrixgui_eigen INTERFACE)
target_include_directories(matrixgui_eigen SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/eigen)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(core)

if(BUILD_CLI OR BUILD_TESTS)
    add_subdirectory(cli)
endif()

if(BUILD_GUI)
    find_package(QT NAMES Qt6 Qt5 COMPONENTS Gui Widgets Concurrent Charts REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Gui Widgets Qml QuickWidgets Concurrent Charts REQUIRED)

    add_subdirectory(contrib/qt-dark-theme)

    add_subdirectory(gui_common)
    add_subdirectory(gui)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
