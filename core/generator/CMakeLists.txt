add_library(matrixgui_generator ${MATRIXGUI_LIB_TYPE}
    trainer.cpp
)

target_link_libraries(matrixgui_generator PUBLIC
    matrixgui_base
    matrixgui_nn
    matrixgui_matrix
    matrixgui_contrib
)
