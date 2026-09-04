add_library(matrixgui_core_lib ${MATRIXGUI_LIB_TYPE}
    stats.cpp
    stream_format.cpp
    text.cpp
)

target_link_libraries(matrixgui_core_lib PUBLIC
    matrixgui_base
)
