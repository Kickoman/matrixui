cmake_minimum_required(VERSION 3.16)
project(QtDarkTheme VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Widgets)

add_library(QtDarkTheme STATIC
    src/QtDarkTheme.cpp
)

target_include_directories(QtDarkTheme PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(QtDarkTheme PUBLIC
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Widgets
)

if(${QT_VERSION_MAJOR} EQUAL 5)
    qt5_add_resources(QDARKTHEME_RESOURCES generated/qdarktheme.qrc)
else()
    qt_add_resources(QDARKTHEME_RESOURCES generated/qdarktheme.qrc)
endif()

target_sources(QtDarkTheme PRIVATE ${QDARKTHEME_RESOURCES})
