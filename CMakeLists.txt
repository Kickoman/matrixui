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
option(BUILD_TESTS "Build unit tests" ON)

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
    core/words/config.cpp
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
    core/words/queries.cpp

    # Words: diagnostics and console rendering
    core/words/diagnostics/sampler_checks.cpp
    core/words/diagnostics/negative_sampler_checks.cpp
    core/words/diagnostics/model_checks.cpp
    core/words/diagnostics/gradient_checks.cpp
    core/words/report/format.cpp
    core/words/report/inspect_report.cpp
    core/words/report/query_report.cpp
    core/words/report/evaluate_report.cpp
    core/words/report/train_report.cpp
    core/words/report/diagnostics_report.cpp

    # Other
    matrix/matrix.cpp
    png/pngreader.cpp
)

include_directories(eigen)

# One compilation of the framework-free core, shared by every executable and by
# the test binary. Previously CORE_SOURCES was compiled separately into each of
# the four targets. Nothing here uses Qt, so the autogen steps are turned off.
add_library(matrixgui_core STATIC ${CORE_SOURCES})
target_include_directories(matrixgui_core PUBLIC contrib)
set_target_properties(matrixgui_core PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

if(BUILD_GUI)
    if(${QT_VERSION_MAJOR} EQUAL 5)
        qt5_add_resources(FONTS gui/fonts/fonts.qrc)
    else()
        qt_add_resources(FONTS gui/fonts/fonts.qrc)
    endif()

    add_executable(
        ${PROJECT_NAME}
        ${GUI_SOURCES}
        ${GUI_COMMON_SOURCES}
        gui/main.cpp
    )

    target_sources(${PROJECT_NAME} PRIVATE ${FONTS})

    target_link_libraries(${PROJECT_NAME}
        matrixgui_core

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
        core/main_headless.cpp
    )
    target_link_libraries(${PROJECT_NAME}_headless matrixgui_core)

    add_executable(
        ${PROJECT_NAME}_gan
        core/main_headless_gan.cpp
    )
    target_link_libraries(${PROJECT_NAME}_gan matrixgui_core)

    # words_cli is deliberately NOT in CORE_SOURCES: that list is compiled into
    # the Qt GUI too, and CLI11 has no business being there.
    add_executable(
        ${PROJECT_NAME}_words
        core/main_words.cpp
        core/words_cli/commands.cpp
    )
    target_link_libraries(${PROJECT_NAME}_words matrixgui_core)

    target_include_directories(${PROJECT_NAME}_headless PUBLIC contrib)
    target_include_directories(${PROJECT_NAME}_gan PUBLIC contrib)
    target_include_directories(${PROJECT_NAME}_words PUBLIC contrib)
endif()

if(BUILD_TESTS)
    enable_testing()

    set(
        TEST_SOURCES
        tests/main.cpp
        tests/words/vocabulary_test.cpp
        tests/words/corpus_test.cpp
        tests/words/embeddings_test.cpp
        tests/words/subsampler_test.cpp
        tests/words/windowsampler_test.cpp
        tests/words/negativesampler_test.cpp
        tests/words/model_test.cpp
        tests/words/similarity_test.cpp
        tests/words/expressions_test.cpp
        tests/words/evaluate_test.cpp
        tests/words/golden_print_test.cpp
        tests/words/trainer_test.cpp
        tests/words/diagnostics_test.cpp
        tests/words/config_test.cpp
        tests/words/inspect_test.cpp
        tests/words/queries_test.cpp
    )

    add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
    target_link_libraries(${PROJECT_NAME}_tests matrixgui_core)
    target_include_directories(${PROJECT_NAME}_tests PUBLIC contrib)
    set_target_properties(${PROJECT_NAME}_tests PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

    add_test(NAME unit COMMAND ${PROJECT_NAME}_tests)
endif()
