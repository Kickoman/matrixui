add_executable(MatrixGui_gan
    main.cpp
)

target_link_libraries(MatrixGui_gan PRIVATE
    matrixgui_generator
    matrixgui_nn
    matrixgui_png
    matrixgui_matrix
    matrixgui_core_lib
    matrixgui_contrib
)
