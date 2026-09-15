add_library(matrixgui_functions ${MATRIXGUI_LIB_TYPE}
    applier.cpp
)

target_link_libraries(matrixgui_functions PUBLIC
    matrixgui_base
    matrixgui_core_lib
    matrixgui_contrib
)
