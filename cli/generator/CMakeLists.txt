add_library(matrixgui_cli_generator ${MATRIXGUI_LIB_TYPE}
    commands.cpp
)

target_link_libraries(matrixgui_cli_generator
    PUBLIC
        matrixgui_base
        matrixgui_generator
        matrixgui_contrib
    PRIVATE
        matrixgui_cli_lib
        matrixgui_nn
        matrixgui_png
        matrixgui_matrix
        matrixgui_core_lib
)

if(BUILD_CLI)
    add_executable(MatrixGui_gan
        main.cpp
    )

    target_link_libraries(MatrixGui_gan PRIVATE
        matrixgui_cli_generator
        matrixgui_contrib
    )
endif()
