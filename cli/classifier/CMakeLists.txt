add_library(matrixgui_cli_classifier ${MATRIXGUI_LIB_TYPE}
    commands.cpp
)

target_link_libraries(matrixgui_cli_classifier
    PUBLIC
        matrixgui_base
        matrixgui_classifier
        matrixgui_nn
        matrixgui_contrib
    PRIVATE
        matrixgui_cli_lib
        matrixgui_png
        matrixgui_matrix
        matrixgui_core_lib
)

if(BUILD_CLI)
    add_executable(MatrixGui_headless
        main.cpp
    )

    target_link_libraries(MatrixGui_headless PRIVATE
        matrixgui_cli_classifier
        matrixgui_cli_lib
        matrixgui_contrib
    )
endif()
