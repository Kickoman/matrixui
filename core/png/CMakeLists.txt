add_library(matrixgui_png ${MATRIXGUI_LIB_TYPE}
    pngreader.cpp
)

target_link_libraries(matrixgui_png PUBLIC
    matrixgui_base
    matrixgui_matrix
)
