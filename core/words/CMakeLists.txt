add_library(matrixgui_words ${MATRIXGUI_LIB_TYPE}
    config.cpp

    data/inspect.cpp
    data/vocabulary.cpp
    data/corpus.cpp
    data/embeddings.cpp

    train/subsampler.cpp
    train/negativesampler.cpp
    train/model.cpp
    train/trainer.cpp

    query/similarity.cpp
    query/expressions.cpp
    query/queries.cpp
    query/evaluate.cpp

    report/inspect_report.cpp
    report/query_report.cpp
    report/evaluate_report.cpp
    report/train_report.cpp
)

target_link_libraries(matrixgui_words PUBLIC
    matrixgui_base
    matrixgui_core_lib
    matrixgui_contrib
)
