add_library(matrixgui_cli_words ${MATRIXGUI_LIB_TYPE}
    commands.cpp
)

target_link_libraries(matrixgui_cli_words PUBLIC
    matrixgui_base
    matrixgui_words
)

if(BUILD_CLI)
    add_executable(MatrixGui_words
        main.cpp
    )

    target_link_libraries(MatrixGui_words PRIVATE
        matrixgui_cli_words
        matrixgui_contrib
    )
endif()
