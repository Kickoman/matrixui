add_library(matrixgui_nn ${MATRIXGUI_LIB_TYPE}
    layers.cpp
    neural_network.cpp
    neural_network_loader.cpp
    neural_network_applier.cpp
    dataset.cpp
    directory_dataset.cpp
)

target_link_libraries(matrixgui_nn
    PUBLIC
        matrixgui_base
        matrixgui_matrix
        matrixgui_contrib
    PRIVATE
        matrixgui_core_lib
)
