add_library(matrixgui_cli_serving ${MATRIXGUI_LIB_TYPE}
    commands.cpp
    server.cpp
)

target_link_libraries(matrixgui_cli_serving PUBLIC
    matrixgui_base
    matrixgui_serving
    matrixgui_contrib
    Threads::Threads
)

if(BUILD_CLI)
    add_executable(MatrixGui_models
        main.cpp
    )

    target_link_libraries(MatrixGui_models PRIVATE
        matrixgui_cli_serving
        matrixgui_contrib
    )
endif()
