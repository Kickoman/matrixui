cmake_minimum_required(VERSION 3.14)

project(MatrixGui LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Qt options
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Gui Widgets Concurrent Charts REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Gui Widgets Qml QuickWidgets Concurrent Charts REQUIRED)

OPTION(USE_EIGEN "Use Eigen matrix" ON) # Enabled by default
IF(USE_EIGEN)
    ADD_DEFINITIONS(-DUSE_EIGEN)
ENDIF(USE_EIGEN)


set(SOURCES
    main.cpp
    main_window.cpp
    advanced_terminal.cpp
    matrix.cpp
    time_chart.cpp
    digit_chart.cpp

    digits_recognizer.cpp
    neural_network.cpp
    pngreader.cpp
    digits_runner.cpp
)

include_directories(eigen)

add_executable(
    ${PROJECT_NAME}
    ${SOURCES}
)


target_link_libraries(${PROJECT_NAME}
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Concurrent
    Qt${QT_VERSION_MAJOR}::Charts
)
