add_executable(MatrixGui_tests
    main.cpp

    words/vocabulary_test.cpp
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

    lib/file_stream_test.cpp
    lib/stream_format_test.cpp
    lib/text_test.cpp
    lib/stats_test.cpp
)

target_link_libraries(MatrixGui_tests PRIVATE
    matrixgui_words
    matrixgui_core_lib
    matrixgui_contrib
)

add_test(NAME unit
    COMMAND MatrixGui_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
