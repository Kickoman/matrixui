#include "cli/words/commands.h"

#include "core/lib/file_stream.h"

#include "core/words/error.h"

#include "core/words/data/corpus.h"
#include "core/words/data/embeddings.h"
#include "core/words/query/evaluate.h"
#include "core/words/query/queries.h"
#include "core/words/report/evaluate_report.h"
#include "core/words/report/inspect_report.h"
#include "core/words/report/query_report.h"
#include "core/words/query/similarity.h"
#include "core/words/data/inspect.h"
#include "core/words/train/trainer.h"
#include "core/words/data/vocabulary.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <ostream>

namespace WordsCli {

namespace {

void PrintVocabularyInfo(std::ostream& out, const Words::Vocabulary& vocabulary) {
    out << "Size: " << vocabulary.getSize() << std::endl;
    out << "First 10 ids:" << std::endl;
    for (Words::TWordId id = 0; id < std::min<Words::TWordId>(10, vocabulary.getSize()); ++id) {
        out << "  - " << vocabulary.getWord(id) << std::endl;
    }
}

void PrintCorpusInfo(std::ostream& out, const Words::TCorpus& corpus) {
    out << "Corpus size: " << corpus.size() << std::endl;
    out << "First 10 tokens: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, corpus.size()); ++i) {
        out << corpus[i] << " ";
    }
    out << std::endl;
}

Words::Vocabulary ReadVocabulary(const std::filesystem::path& path) {
    return Io::ReadFile(path, [](std::istream& in) { return Words::Vocabulary::Load(in); },
                        std::ios::binary);
}

Words::EmbeddingIndex ReadEmbeddings(const std::filesystem::path& path) {
    return Io::ReadFile(path, [](std::istream& in) { return Words::EmbeddingIndex::Load(in); },
                        std::ios::binary);
}

}  // namespace

namespace {

void RunInspect(std::ostream& out, const InspectOptions& options) {
    const auto statistics = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::InspectDump(in, options.topN);
    });
    Words::PrintCorpusStatistics(out, options.input.string(), statistics);
}

void RunBuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options) {
    const auto vocabulary = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::Vocabulary::Build(in, options.minCount);
    });
    PrintVocabularyInfo(out, vocabulary);
    Io::WriteFile(options.output,
                  [&](std::ostream& file) { Words::Vocabulary::Save(file, vocabulary); },
                  std::ios::binary);
}

void RunLoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options) {
    PrintVocabularyInfo(out, ReadVocabulary(options.input));
}

void RunBuildCorpus(std::ostream& out, const BuildCorpusOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto corpus = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::EncodeCorpus(in, vocabulary);
    });
    PrintCorpusInfo(out, corpus);
    Io::WriteFile(options.output,
                  [&](std::ostream& file) { Words::SaveCorpus(file, corpus); },
                  std::ios::binary);
}

void RunLoadCorpus(std::ostream& out, const LoadCorpusOptions& options) {
    PrintCorpusInfo(out, Io::ReadFile(options.input,
                                      [](std::istream& in) { return Words::LoadCorpus(in); },
                                      std::ios::binary));
}

void RunTrain(std::ostream& out, const TrainOptions& options) {
    auto vocabulary = std::make_shared<const Words::Vocabulary>(ReadVocabulary(options.vocabulary));
    auto corpus = std::make_shared<const Words::TCorpus>(
        Io::ReadFile(options.corpus, [](std::istream& in) { return Words::LoadCorpus(in); },
                     std::ios::binary));

    Words::Trainer trainer;
    trainer.setVocabulary(vocabulary);
    trainer.setCorpus(corpus);
    trainer.setModelConfig(options.config.model);
    trainer.setSamplingConfig(options.config.sampling);
    trainer.setOutputStream(&out);

    out << "vocab " << vocabulary->getSize()
        << ", corpus " << corpus->size()
        << ", model "
        << 2 * vocabulary->getSize() * options.config.model.dim * sizeof(Words::TFloat) / (1024 * 1024)
        << " MB\n";

    trainer.train(options.config.train);

    Io::WriteFile(options.output,
                  [&](std::ostream& file) {
                      Words::Embeddings::Save(file, trainer.getInputEmbeddings());
                  },
                  std::ios::binary);
    out << "saved embeddings to " << options.output << '\n';
}

void RunNeighbours(std::ostream& out, const NeighboursOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    out << "words " << index.getWords() << ", dim " << index.getDim() << "\n\n";

    if (options.word.empty()) {
        Words::PrintBatteryReport(out, vocabulary, Words::RunDefaultBattery(vocabulary, index));
    } else {
        Words::PrintNeighbourReport(
            out, vocabulary, Words::QueryNeighbours(vocabulary, index, options.word, options.count));
    }
}

void RunEvaluate(std::ostream& out, const EvaluateOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    out << "words " << index.getWords() << ", dim " << index.getDim() << '\n';

    if (!options.analogies.empty()) {
        out << "\n=== Analogies ===\n";
        Words::PrintAnalogyReport(
            out,
            Io::ReadFile(options.analogies, [&](std::istream& in) {
                return Words::EvaluateAnalogies(
                    vocabulary, index, in, options.restrictTo, options.threads);
            }));
    }

    if (!options.similarity.empty()) {
        out << "\n=== Similarity ===\n";
        Words::PrintSimilarityReport(
            out,
            options.similarity.filename().string(),
            Io::ReadFile(options.similarity, [&](std::istream& in) {
                return Words::EvaluateSimilarity(vocabulary, index, in, options.scoreColumn);
            }));
    }
}

void RunExpression(std::ostream& out, const ExpressionOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintExpressionReport(
        out, vocabulary,
        Words::QueryExpression(vocabulary, index, options.expression, options.count));
}

void RunOddOne(std::ostream& out, const OddOneOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintOddOneOutReport(
        out, vocabulary, Words::QueryOddOneOut(vocabulary, index, options.words));
}

void RunAxis(std::ostream& out, const AxisOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintAxisReport(
        out, vocabulary,
        Words::QueryAxis(vocabulary, index, options.axis, options.words,
                         options.restrictTo, options.count));
}

template <typename Body>
int Guarded(std::ostream& err, Body&& body) {
    try {
        body();
        return kSuccess;
    } catch (const Io::Error& error) {
        err << "error: " << error.what() << '\n';
        return kFailure;
    } catch (const Words::Error& error) {
        err << "error: " << error.what() << '\n';
        return kFailure;
    } catch (const std::exception& error) {
        err << "internal error: " << error.what() << '\n';
        return kInternalError;
    }
}

}  // namespace

int Inspect(std::ostream& out, std::ostream& err, const InspectOptions& options) {
    return Guarded(err, [&] { RunInspect(out, options); });
}

int BuildVocabulary(std::ostream& out, std::ostream& err, const BuildVocabularyOptions& options) {
    return Guarded(err, [&] { RunBuildVocabulary(out, options); });
}

int LoadVocabulary(std::ostream& out, std::ostream& err, const LoadVocabularyOptions& options) {
    return Guarded(err, [&] { RunLoadVocabulary(out, options); });
}

int BuildCorpus(std::ostream& out, std::ostream& err, const BuildCorpusOptions& options) {
    return Guarded(err, [&] { RunBuildCorpus(out, options); });
}

int LoadCorpus(std::ostream& out, std::ostream& err, const LoadCorpusOptions& options) {
    return Guarded(err, [&] { RunLoadCorpus(out, options); });
}

int Train(std::ostream& out, std::ostream& err, const TrainOptions& options) {
    return Guarded(err, [&] { RunTrain(out, options); });
}

int Neighbours(std::ostream& out, std::ostream& err, const NeighboursOptions& options) {
    return Guarded(err, [&] { RunNeighbours(out, options); });
}

int Evaluate(std::ostream& out, std::ostream& err, const EvaluateOptions& options) {
    return Guarded(err, [&] { RunEvaluate(out, options); });
}

int Expression(std::ostream& out, std::ostream& err, const ExpressionOptions& options) {
    return Guarded(err, [&] { RunExpression(out, options); });
}

int OddOne(std::ostream& out, std::ostream& err, const OddOneOptions& options) {
    return Guarded(err, [&] { RunOddOne(out, options); });
}

int Axis(std::ostream& out, std::ostream& err, const AxisOptions& options) {
    return Guarded(err, [&] { RunAxis(out, options); });
}

}  // namespace WordsCli
