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

    add_subdirectory(contrib/qt-dark-theme)
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
    gui/lib/theme.cpp
    gui/lib/mode_factory.cpp
    gui/lib/mode_settings.cpp
    gui/lib/mode_widget.cpp
    gui/lib/mode_controller.cpp
    gui/lib/network_create_dialog.cpp
    gui/lib/new_mode_widget.cpp
    gui/lib/mode_container_widget.cpp

    gui/classifier/digits_classifier_controller.cpp
    gui/classifier/digits_classifier_mode_widget.cpp
    gui/classifier/learning_config_widget.cpp

    gui/recognizer/digits_recognizer_mode_controller.cpp
    gui/recognizer/digits_recognizer_mode_widget.cpp
    gui/recognizer/digit_input_preprocess.cpp

    gui/generator/digits_generator_controller.cpp
    gui/generator/digits_generator_mode_widget.cpp
    gui/generator/gan_config_widget.cpp

    gui/main_controller.cpp
    gui/main_window.cpp
)

set(
    GUI_COMMON_SOURCES
    gui_common/add_tab_bar.cpp
    gui_common/add_tab_widget.cpp
    gui_common/advanced_terminal.cpp
    gui_common/time_chart.cpp
    gui_common/digit_chart.cpp
    gui_common/paint_scene.cpp
)

set(
    CORE_SOURCES
    core/lib/layers.cpp
    core/lib/neural_network.cpp
    core/lib/neural_network_loader.cpp
    core/lib/neural_network_applier.cpp
    core/lib/dataset.cpp
    core/lib/directory_dataset.cpp

    # Classifier
    core/classifier/trainer.cpp

    # Generator
    core/generator/trainer.cpp

    # Words
    core/words/utils/inspect.cpp
    core/words/vocabulary.cpp
    core/words/corpus.cpp
    core/words/subsampler.cpp
    core/words/negativesampler.cpp
    core/words/embeddings.cpp
    core/words/model.cpp
    core/words/trainer.cpp
    core/words/similarity.cpp
    core/words/evaluate.cpp
    core/words/expressions.cpp

    # Other
    matrix/matrix.cpp
    png/pngreader.cpp
)

include_directories(eigen)

if(BUILD_GUI)
    if(${QT_VERSION_MAJOR} EQUAL 5)
        qt5_add_resources(FONTS gui/fonts/fonts.qrc)
    else()
        qt_add_resources(FONTS gui/fonts/fonts.qrc)
    endif()

    add_executable(
        ${PROJECT_NAME}
        ${CORE_SOURCES}
        ${GUI_SOURCES}
        ${GUI_COMMON_SOURCES}
        gui/main.cpp
    )

    target_sources(${PROJECT_NAME} PRIVATE ${FONTS})

    target_link_libraries(${PROJECT_NAME}
        Qt${QT_VERSION_MAJOR}::Gui
        Qt${QT_VERSION_MAJOR}::Widgets
        Qt${QT_VERSION_MAJOR}::Concurrent
        Qt${QT_VERSION_MAJOR}::Charts

        QtDarkTheme
    )

    target_include_directories(${PROJECT_NAME} PUBLIC contrib)

endif()

if(BUILD_CLI)
    add_executable(
        ${PROJECT_NAME}_headless
        ${CORE_SOURCES}
        core/main_headless.cpp
    )

    add_executable(
        ${PROJECT_NAME}_gan
        ${CORE_SOURCES}
        core/main_headless_gan.cpp
    )

    add_executable(
        ${PROJECT_NAME}_words
        ${CORE_SOURCES}
        core/main_words.cpp
    )

    target_include_directories(${PROJECT_NAME}_headless PUBLIC contrib)
    target_include_directories(${PROJECT_NAME}_gan PUBLIC contrib)
    target_include_directories(${PROJECT_NAME}_words PUBLIC contrib)
endif()
