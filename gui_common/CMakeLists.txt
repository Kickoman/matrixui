add_library(matrixgui_gui_common ${MATRIXGUI_LIB_TYPE}
    add_tab_bar.cpp
    add_tab_widget.cpp
    advanced_terminal.cpp
    time_chart.cpp
    digit_chart.cpp
    paint_scene.cpp
)

set_target_properties(matrixgui_gui_common PROPERTIES AUTOMOC ON)

target_link_libraries(matrixgui_gui_common PUBLIC
    matrixgui_base
    Qt${QT_VERSION_MAJOR}::Gui
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Charts
)
