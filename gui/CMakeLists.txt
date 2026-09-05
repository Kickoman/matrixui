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
