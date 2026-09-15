#include "cli/words/commands.h"
#include "cli/words/options.h"

#include <CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"Words embedder"};
    app.require_subcommand(1);

    int exitCode = WordsCli::kSuccess;

    WordsCli::InspectOptions inspect;
    auto* inspectCmd = app.add_subcommand("inspect", "Summarise a raw text dump");
    inspectCmd->add_option("--input-file", inspect.input, "Raw text file")
        ->required()->check(CLI::ExistingFile);
    inspectCmd->add_option("--top", inspect.topN, "How many frequent words to list")
        ->capture_default_str();
    inspectCmd->callback([&] {
        exitCode = WordsCli::Inspect(std::cout, std::cerr, inspect);
    });

    WordsCli::BuildVocabularyOptions buildVocabulary;
    auto* buildVocabularyCmd = app.add_subcommand("buildvoc", "Build vocabulary");
    buildVocabularyCmd->add_option("--input-file", buildVocabulary.input, "Prepared input file path")
        ->required()->check(CLI::ExistingFile);
    buildVocabularyCmd->add_option("--output-file", buildVocabulary.output, "Output file for built vocabulary")
        ->required();
    buildVocabularyCmd->add_option("--min-count", buildVocabulary.minCount, "Drop words below this count")
        ->capture_default_str();
    buildVocabularyCmd->add_option("--prune-threshold", buildVocabulary.pruneThreshold,
                                   "Prune rare words when distinct words exceed this (0 = never)")
        ->capture_default_str();
    buildVocabularyCmd->callback([&] {
        exitCode = WordsCli::BuildVocabulary(std::cout, std::cerr, buildVocabulary);
    });

    WordsCli::LoadVocabularyOptions loadVocabulary;
    auto* loadVocabularyCmd = app.add_subcommand("loadvoc", "Load vocabulary");
    loadVocabularyCmd->add_option("--input-file", loadVocabulary.input, "Built vocabulary file")
        ->required()->check(CLI::ExistingFile);
    loadVocabularyCmd->callback([&] {
        exitCode = WordsCli::LoadVocabulary(std::cout, std::cerr, loadVocabulary);
    });

    WordsCli::BuildCorpusOptions buildCorpus;
    auto* buildCorpusCmd = app.add_subcommand("buildcor", "Build corpus info");
    buildCorpusCmd->add_option("--input-file", buildCorpus.input, "Prepared corpus file path")
        ->required()->check(CLI::ExistingFile);
    buildCorpusCmd->add_option("--output-file", buildCorpus.output, "Output file for built corpus")
        ->required();
    buildCorpusCmd->add_option("--vocabulary", buildCorpus.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    buildCorpusCmd->callback([&] {
        exitCode = WordsCli::BuildCorpus(std::cout, std::cerr, buildCorpus);
    });

    WordsCli::LoadCorpusOptions loadCorpus;
    auto* loadCorpusCmd = app.add_subcommand("loadcor", "Load corpus info");
    loadCorpusCmd->add_option("--input-file", loadCorpus.input, "Built corpus file path")
        ->required()->check(CLI::ExistingFile);
    loadCorpusCmd->callback([&] {
        exitCode = WordsCli::LoadCorpus(std::cout, std::cerr, loadCorpus);
    });

    WordsCli::TrainOptions train;
    auto* trainCmd = app.add_subcommand("train", "Train SGNS embeddings");
    trainCmd->add_option("--vocabulary", train.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    trainCmd->add_option("--corpus", train.corpus, "Built corpus path")
        ->required()->check(CLI::ExistingFile);
    trainCmd->add_option("--output-file", train.output, "Output file for embeddings")
        ->required();
    trainCmd->add_option("--dim", train.config.model.dim, "Embedding dimension")->capture_default_str();
    trainCmd->add_option("--negatives", train.config.model.negatives, "Negatives per pair")->capture_default_str();
    trainCmd->add_option("--window", train.config.sampling.window, "Max window radius")->capture_default_str();
    trainCmd->add_option("--epochs", train.config.train.epochs, "Epochs")->capture_default_str();
    trainCmd->add_option("--sample", train.config.sampling.sample, "Subsampling threshold")->capture_default_str();
    trainCmd->add_option("--lr", train.config.model.initialLearningRate, "Initial learning rate")->capture_default_str();
    trainCmd->add_option("--threads", train.config.train.threads, "Worker threads (0 = auto)")->capture_default_str();
    trainCmd->add_option("--corpus-storage", train.corpusStorage, "Corpus storage: auto, mmap or load")
        ->check(CLI::IsMember({"auto", "mmap", "load"}))->capture_default_str();
    trainCmd->callback([&] {
        exitCode = WordsCli::Train(std::cout, std::cerr, train);
    });

    WordsCli::NeighboursOptions neighbours;
    auto* neighboursCmd = app.add_subcommand("neighbours", "Show nearest neighbours");
    neighboursCmd->add_option("--vocabulary", neighbours.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    neighboursCmd->add_option("--embeddings", neighbours.embeddings, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    neighboursCmd->add_option("--word", neighbours.word, "Query word (empty runs a default battery)");
    neighboursCmd->add_option("--count", neighbours.count, "How many neighbours")->capture_default_str();
    neighboursCmd->callback([&] {
        exitCode = WordsCli::Neighbours(std::cout, std::cerr, neighbours);
    });

    WordsCli::EvaluateOptions evaluate;
    auto* evaluateCmd = app.add_subcommand("evaluate", "Evaluate embeddings");
    evaluateCmd->add_option("--vocabulary", evaluate.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    evaluateCmd->add_option("--embeddings", evaluate.embeddings, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    evaluateCmd->add_option("--analogies", evaluate.analogies, "questions-words.txt")
        ->check(CLI::ExistingFile);
    evaluateCmd->add_option("--similarity", evaluate.similarity, "WordSim/SimLex file")
        ->check(CLI::ExistingFile);
    evaluateCmd->add_option("--score-column", evaluate.scoreColumn, "0-based score column")->capture_default_str();
    evaluateCmd->add_option("--restrict-to", evaluate.restrictTo, "Search top-N words only (0 = all)")->capture_default_str();
    evaluateCmd->add_option("--threads", evaluate.threads, "Worker threads (0 = auto)")->capture_default_str();
    evaluateCmd->callback([&] {
        exitCode = WordsCli::Evaluate(std::cout, std::cerr, evaluate);
    });

    WordsCli::ExpressionOptions expression;
    auto* expressionCmd = app.add_subcommand("expression", "Evaluate a vector expression");
    expressionCmd->add_option("expression", expression.expression, "e.g. \"king - man + woman\"")->required();
    expressionCmd->add_option("--vocabulary", expression.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    expressionCmd->add_option("--embeddings", expression.embeddings, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    expressionCmd->add_option("--count", expression.count, "How many results")->capture_default_str();
    expressionCmd->callback([&] {
        exitCode = WordsCli::Expression(std::cout, std::cerr, expression);
    });

    WordsCli::OddOneOptions oddOne;
    auto* oddOneCmd = app.add_subcommand("oddone", "Find the odd word out");
    oddOneCmd->add_option("words", oddOne.words, "e.g. \"breakfast cereal lunch dinner\"")->required();
    oddOneCmd->add_option("--vocabulary", oddOne.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    oddOneCmd->add_option("--embeddings", oddOne.embeddings, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    oddOneCmd->callback([&] {
        exitCode = WordsCli::OddOne(std::cout, std::cerr, oddOne);
    });

    WordsCli::AxisOptions axis;
    auto* axisCmd = app.add_subcommand("axis", "Project words onto a semantic axis");
    axisCmd->add_option("axis", axis.axis, "e.g. \"good - bad\"")->required();
    axisCmd->add_option("--words", axis.words, "Words to project (empty scans the vocabulary)");
    axisCmd->add_option("--vocabulary", axis.vocabulary, "Built vocabulary path")
        ->required()->check(CLI::ExistingFile);
    axisCmd->add_option("--embeddings", axis.embeddings, "Trained embeddings")
        ->required()->check(CLI::ExistingFile);
    axisCmd->add_option("--restrict-to", axis.restrictTo, "Scan top-N words (0 = all)")->capture_default_str();
    axisCmd->add_option("--count", axis.count, "How many per end")->capture_default_str();
    axisCmd->callback([&] {
        exitCode = WordsCli::Axis(std::cout, std::cerr, axis);
    });

    // CLI11_PARSE cannot be used here: the subcommand callbacks run inside
    // parse(), and the macro's catch covers only CLI::ParseError.
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    return exitCode;
}
