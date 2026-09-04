add_library(matrixgui_cli_lib ${MATRIXGUI_LIB_TYPE}
    argv_scan.cpp
    png_reader.cpp
)

target_link_libraries(matrixgui_cli_lib
    PUBLIC
        matrixgui_base
        matrixgui_matrix
        matrixgui_contrib
    PRIVATE
        matrixgui_png
        matrixgui_core_lib
)
