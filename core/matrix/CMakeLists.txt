add_library(matrixgui_matrix ${MATRIXGUI_LIB_TYPE}
    matrix.cpp
)

target_link_libraries(matrixgui_matrix PUBLIC
    matrixgui_base
    matrixgui_eigen
)
