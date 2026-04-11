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

option(BUILD_GUI "Build GUI version" ON)
option(BUILD_CLI "Build CLI version" ON)
option(BUILD_READER "Build Reader version" ON)

# Qt options
if (BUILD_GUI)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)

    find_package(QT NAMES Qt6 Qt5 COMPONENTS Gui Widgets Concurrent Charts REQUIRED)
    find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Gui Widgets Qml QuickWidgets Concurrent Charts REQUIRED)
endif()

option(USE_EIGEN "Use Eigen matrix" ON) # Enabled by default
if(USE_EIGEN)
    message("Using Eigen't matrix implementation")
    add_definitions(-DUSE_EIGEN)
    # Hide warnings for eigen
    set_directory_properties(PATH eigen PROPERTIES
        COMPILE_OPTIONS "-w"  # For GCC/Clang
        # COMPILE_OPTIONS "/w"  # For MSVC
    )
endif(USE_EIGEN)

set(GUI_SOURCES
    gui/digits_runner.cpp
    gui/main_window.cpp
    gui/network_create_dialog.cpp
    gui/learning_config_widget.cpp
)

set(
    GUI_COMMON_SOURCES
    gui_common/advanced_terminal.cpp
    gui_common/time_chart.cpp
    gui_common/digit_chart.cpp
)

set(SOURCES
    matrix/matrix.cpp
    core/layers.cpp
    core/neural_network_loader.cpp
    core/neural_network_applier.cpp
    png/pngreader.cpp
    core/dataset.cpp
    core/directory_dataset.cpp
    core/trainer.cpp
)

set(READER_SOURCES
    reader/main_reader.cpp
    reader/main_window.cpp
    reader/paint_scene.cpp
    reader/digits_tester.cpp
    reader/digit_input_preprocess.cpp
)

include_directories(eigen)

if(BUILD_GUI)
    add_executable(
        ${PROJECT_NAME}
        ${GUI_SOURCES}
        ${GUI_COMMON_SOURCES}
        ${MATRIX_SOURCES}
        ${SOURCES}
        gui/main.cpp
    )

    target_link_libraries(${PROJECT_NAME}
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Concurrent
        Qt${QT_VERSION_MAJOR}::Charts
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        core/
        gui/
        gui_common/
        matrix/
        png/
    )
endif()

if(BUILD_CLI)
    add_executable(
        ${PROJECT_NAME}_headless
        ${MATRIX_SOURCES}
        ${SOURCES}
        core/main_headless.cpp
    )

    target_include_directories(${PROJECT_NAME}_headless PRIVATE
        core/
        matrix/
        png/
    )
endif()

if(BUILD_READER)
    add_executable(
        ${PROJECT_NAME}_reader
        ${MATRIX_SOURCES}
        ${SOURCES}
        ${GUI_COMMON_SOURCES}
        ${READER_SOURCES}
    )

    target_link_libraries(${PROJECT_NAME}_reader
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Widgets
        # Qt${QT_VERSION_MAJOR}::Concurrent
        Qt${QT_VERSION_MAJOR}::Charts
    )

    target_include_directories(
        ${PROJECT_NAME}_reader PRIVATE
        core/
        matrix/
        png/
        gui_common/
        reader/
    )
endif()
