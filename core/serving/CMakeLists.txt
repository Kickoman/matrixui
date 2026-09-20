add_library(matrixgui_serving ${MATRIXGUI_LIB_TYPE}
    manifest.cpp
    snapshot.cpp
    registry.cpp
    build.cpp
    loaded_model.cpp
    model_loader.cpp
)

target_link_libraries(matrixgui_serving
    PUBLIC
        matrixgui_base
        matrixgui_nn
        matrixgui_contrib
    PRIVATE
        matrixgui_core_lib
)
