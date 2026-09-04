#include "core/words_cli/commands.h"

#include "core/lib/file_stream.h"

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

void Inspect(std::ostream& out, const InspectOptions& options) {
    const auto statistics = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::InspectDump(in, options.topN);
    });
    Words::PrintCorpusStatistics(out, options.input.string(), statistics);
}

void BuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options) {
    const auto vocabulary = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::Vocabulary::Build(in, options.minCount);
    });
    PrintVocabularyInfo(out, vocabulary);
    Io::WriteFile(options.output,
                  [&](std::ostream& file) { Words::Vocabulary::Save(file, vocabulary); },
                  std::ios::binary);
}

void LoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options) {
    PrintVocabularyInfo(out, ReadVocabulary(options.input));
}

void BuildCorpus(std::ostream& out, const BuildCorpusOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto corpus = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::EncodeCorpus(in, vocabulary);
    });
    PrintCorpusInfo(out, corpus);
    Io::WriteFile(options.output,
                  [&](std::ostream& file) { Words::SaveCorpus(file, corpus); },
                  std::ios::binary);
}

void LoadCorpus(std::ostream& out, const LoadCorpusOptions& options) {
    PrintCorpusInfo(out, Io::ReadFile(options.input,
                                      [](std::istream& in) { return Words::LoadCorpus(in); },
                                      std::ios::binary));
}

void Train(std::ostream& out, const TrainOptions& options) {
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

void Neighbours(std::ostream& out, const NeighboursOptions& options) {
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

void Evaluate(std::ostream& out, const EvaluateOptions& options) {
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

void Expression(std::ostream& out, const ExpressionOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintExpressionReport(
        out, vocabulary,
        Words::QueryExpression(vocabulary, index, options.expression, options.count));
}

void OddOne(std::ostream& out, const OddOneOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintOddOneOutReport(
        out, vocabulary, Words::QueryOddOneOut(vocabulary, index, options.words));
}

void Axis(std::ostream& out, const AxisOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);

    Words::PrintAxisReport(
        out, vocabulary,
        Words::QueryAxis(vocabulary, index, options.axis, options.words,
                         options.restrictTo, options.count));
}

}  // namespace WordsCli
