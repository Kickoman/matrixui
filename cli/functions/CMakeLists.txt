add_library(matrixgui_cli_functions ${MATRIXGUI_LIB_TYPE}
    commands.cpp
)

target_link_libraries(matrixgui_cli_functions PUBLIC
    matrixgui_base
    matrixgui_functions
    matrixgui_contrib
)

if(BUILD_CLI)
    add_executable(MatrixGui_functions
        main.cpp
    )

    target_link_libraries(MatrixGui_functions PRIVATE
        matrixgui_cli_functions
        matrixgui_cli_lib
        matrixgui_contrib
    )
endif()
