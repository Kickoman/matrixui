if(${QT_VERSION_MAJOR} EQUAL 5)
    qt5_add_resources(FONTS fonts/fonts.qrc)
else()
    qt_add_resources(FONTS fonts/fonts.qrc)
endif()

add_executable(MatrixGui
    lib/theme.cpp
    lib/mode_factory.cpp
    lib/mode_settings.cpp
    lib/mode_widget.cpp
    lib/mode_controller.cpp
    lib/network_create_dialog.cpp
    lib/new_mode_widget.cpp
    lib/mode_container_widget.cpp

    classifier/digits_classifier_controller.cpp
    classifier/digits_classifier_mode_widget.cpp
    classifier/learning_config_widget.cpp

    recognizer/digits_recognizer_mode_controller.cpp
    recognizer/digits_recognizer_mode_widget.cpp
    recognizer/digit_input_preprocess.cpp

    generator/digits_generator_controller.cpp
    generator/digits_generator_mode_widget.cpp
    generator/gan_config_widget.cpp

    words/words_controller.cpp
    words/words_mode_widget.cpp
    words/words_data_tab_widget.cpp
    words/words_explore_tab_widget.cpp
    words/words_evaluate_tab_widget.cpp
    words/words_train_config_widget.cpp

    main_controller.cpp
    main_window.cpp
    main.cpp

    ${FONTS}
)

set_target_properties(MatrixGui PROPERTIES AUTOMOC ON)

# On macOS a Qt app has to be a bundle to behave like an application at all:
# without one it runs as a faceless background process, gets no Dock icon, cannot
# reliably be brought to the foreground, and QSettings keys off an empty bundle
# identifier. This is the one target that escapes CMAKE_RUNTIME_OUTPUT_DIRECTORY --
# it lands in <build>/MatrixGui.app/Contents/MacOS/. The CLI tools, and so every
# tests/golden script, are unaffected.
if(APPLE)
    set_target_properties(MatrixGui PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "MatrixGui"
        MACOSX_BUNDLE_GUI_IDENTIFIER "by.kastus.matrixgui"
        MACOSX_BUNDLE_BUNDLE_VERSION "1.0"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0"
    )
endif()

target_link_libraries(MatrixGui PRIVATE
    matrixgui_gui_common

    matrixgui_classifier
    matrixgui_generator
    matrixgui_words
    matrixgui_nn
    matrixgui_png
    matrixgui_matrix
    matrixgui_core_lib
    matrixgui_contrib

    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Concurrent
    Qt${QT_VERSION_MAJOR}::Charts

    QtDarkTheme
)
