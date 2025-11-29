cmake_minimum_required(VERSION 3.14)

project(MatrixGui LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

OPTION(BUILD_GUI "Build GUI version" ON)
OPTION(BUILD_CLI "Build CLI version" ON)

# Qt options
if (BUILD_GUI)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    find_package(QT NAMES Qt6 Qt5 COMPONENTS Gui Widgets Concurrent Charts REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Gui Widgets Qml QuickWidgets Concurrent Charts REQUIRED)
endif()

set(USE_EIGEN ON)
add_subdirectory(matrix)

set(
    QT_DEPENDANT_SOURCES
    main_window.cpp
    advanced_terminal.cpp
    time_chart.cpp
    digit_chart.cpp
    digits_runner.cpp
    network_create_dialog.cpp
)

set(SOURCES
    digits_recognizer.cpp
    neural_network.cpp
    pngreader.cpp
)

include_directories(eigen)


IF(BUILD_GUI)
    add_executable(
        ${PROJECT_NAME}
        ${QT_DEPENDANT_SOURCES}
        ${MATRIX_SOURCES}
        ${SOURCES}
        main.cpp
    )

   target_link_libraries(${PROJECT_NAME}
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Concurrent
        Qt${QT_VERSION_MAJOR}::Charts
        Matrix
    )
ENDIF()

IF(BUILD_CLI)
    add_executable(
        ${PROJECT_NAME}_headless
        ${MATRIX_SOURCES}
        ${SOURCES}
        main_headless.cpp
    )
    target_link_libraries(
        ${PROJECT_NAME}_headless Matrix
    )
ENDIF()

OPTION(BUILD_MATRIX_BENCHMARK "Build MatrixBenchmark executable" OFF)

if(BUILD_MATRIX_BENCHMARK)
    message("Building MatrixBenchmark with perftest library")

    set(PERFBENCH_BUILD_EXAMPLES OFF CACHE BOOL "Build example benchmarks" FORCE)
    set(PERFBENCH_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)

    add_subdirectory(perftest)

    add_library(matrix_benchmarks OBJECT
        matrix_benchmarks.cpp
        recognition_benchmarks.cpp
        digits_recognizer.cpp
        neural_network.cpp
        pngreader.cpp
    )

    target_link_libraries(matrix_benchmarks perfbench Matrix)
    target_include_directories(matrix_benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen
    )

    target_sources(perfbench_main PRIVATE
        $<TARGET_OBJECTS:matrix_benchmarks>
        $<TARGET_OBJECTS:Matrix>
    )

    target_compile_options(matrix_benchmarks PRIVATE -O3 -DNDEBUG)
endif()
