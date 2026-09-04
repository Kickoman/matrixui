add_library(matrixgui_classifier ${MATRIXGUI_LIB_TYPE}
    trainer.cpp
)

target_link_libraries(matrixgui_classifier PUBLIC
    matrixgui_base
    matrixgui_nn
    matrixgui_contrib
)
