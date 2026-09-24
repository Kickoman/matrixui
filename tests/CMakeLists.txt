add_executable(MatrixGui_tests
    main.cpp

    words/vocabulary_test.cpp
    words/subwords_test.cpp
    words/corpus_test.cpp
    words/embeddings_test.cpp
    words/subsampler_test.cpp
    words/windowsampler_test.cpp
    words/negativesampler_test.cpp
    words/model_test.cpp
    words/similarity_test.cpp
    words/expressions_test.cpp
    words/evaluate_test.cpp
    words/golden_print_test.cpp
    words/trainer_test.cpp
    words/config_test.cpp
    words/inspect_test.cpp
    words/queries_test.cpp

    matrix/matrix_test.cpp

    nn/neural_network_test.cpp
    nn/neural_network_loader_test.cpp
    nn/neural_network_applier_test.cpp

    serving/build_test.cpp
    serving/manifest_test.cpp
    serving/registry_test.cpp
    serving/model_loader_test.cpp

    functions/applier_test.cpp
    functions/expected_csv_test.cpp
    functions/validate_test.cpp
    functions/genetizer_test.cpp

    lib/file_stream_test.cpp
    lib/mapped_file_test.cpp
    lib/stream_format_test.cpp
    lib/text_test.cpp
    lib/sha256_test.cpp
    lib/stats_test.cpp
    lib/rpn_test.cpp
    lib/cache_test.cpp
    lib/rpn_compile_test.cpp

    cli/argv_scan_test.cpp
    cli/json_config_test.cpp
    cli/png_reader_test.cpp
    cli/classifier_commands_test.cpp
    cli/generator_commands_test.cpp
    cli/serving_commands_test.cpp
    cli/serving_handlers_test.cpp
    cli/serving_server_test.cpp
    cli/words_commands_test.cpp
    cli/functions_commands_test.cpp
)

target_link_libraries(MatrixGui_tests PRIVATE
    matrixgui_words
    matrixgui_core_lib
    matrixgui_contrib

    matrixgui_cli_lib
    matrixgui_cli_words
    matrixgui_cli_classifier
    matrixgui_cli_generator
    matrixgui_cli_functions
    matrixgui_cli_serving
    matrixgui_nn
    matrixgui_serving
    matrixgui_png
    matrixgui_matrix
)

add_test(NAME unit
    COMMAND MatrixGui_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
