add_executable(MatrixGui_headless
    main.cpp
)

target_link_libraries(MatrixGui_headless PRIVATE
    matrixgui_classifier
    matrixgui_nn
    matrixgui_png
    matrixgui_matrix
    matrixgui_core_lib
    matrixgui_contrib
)
