#include "cli/words/commands.h"

#include "core/lib/file_stream.h"

#include "core/words/error.h"

#include "core/words/data/corpus.h"
#include "core/words/data/embeddings.h"
#include "core/words/data/subwords.h"
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

void PrintCorpusInfo(std::ostream& out, const Words::Corpus& corpus) {
    out << "Corpus size: " << corpus.size() << std::endl;
    out << "First 10 tokens: ";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, corpus.size()); ++i) {
        out << corpus[i] << " ";
    }
    out << std::endl;
}

Words::CorpusStorage ParseCorpusStorage(const std::string& name) {
    if (name == "auto") {
        return Words::CorpusStorage::Auto;
    }
    if (name == "mmap") {
        return Words::CorpusStorage::Mapped;
    }
    if (name == "load") {
        return Words::CorpusStorage::Loaded;
    }
    throw Words::ConfigError("unknown corpus storage -- expected auto, mmap or load: " + name);
}

Words::Vocabulary ReadVocabulary(const std::filesystem::path& path) {
    return Io::ReadFile(path, [](std::istream& in) { return Words::Vocabulary::Load(in); },
                        std::ios::binary);
}

Words::EmbeddingIndex ReadEmbeddings(const std::filesystem::path& path) {
    return Io::ReadFile(path, [](std::istream& in) { return Words::EmbeddingIndex::Load(in); },
                        std::ios::binary);
}

Words::SubwordVectors ReadSubwords(const std::filesystem::path& path) {
    return Io::ReadFile(path, [](std::istream& in) { return Words::SubwordVectors::Load(in); },
                        std::ios::binary);
}

void RequireMatchingVocabulary(
    const Words::EmbeddingIndex& index, const Words::Vocabulary& vocabulary) {
    if (index.getWords() != vocabulary.getSize()) {
        throw Words::IoError(
            "embeddings hold " + std::to_string(index.getWords())
            + " rows but the vocabulary has " + std::to_string(vocabulary.getSize()) + " words");
    }
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
    Words::VocabularyBuildStats stats;
    const auto vocabulary = Io::ReadFile(options.input, [&](std::istream& in) {
        return Words::Vocabulary::Build(in, options.minCount, options.pruneThreshold, &stats);
    });
    PrintVocabularyInfo(out, vocabulary);
    if (stats.pruneRuns > 0) {
        out << "Pruned " << stats.pruneRuns << " times (final min-reduce "
            << stats.finalMinReduce << ")" << std::endl;
    }
    Io::WriteFile(options.output,
                  [&](std::ostream& file) { Words::Vocabulary::Save(file, vocabulary); },
                  std::ios::binary);
}

void RunLoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options) {
    PrintVocabularyInfo(out, ReadVocabulary(options.input));
}

void RunBuildCorpus(std::ostream& out, const BuildCorpusOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    Io::WriteFile(options.output,
                  [&](std::ostream& file) {
                      Io::ReadFile(options.input, [&](std::istream& in) {
                          Words::EncodeCorpusToStream(in, vocabulary, file);
                      });
                  },
                  std::ios::binary);
    PrintCorpusInfo(out, Words::Corpus::Open(options.output));
}

void RunLoadCorpus(std::ostream& out, const LoadCorpusOptions& options) {
    PrintCorpusInfo(out, Words::Corpus::Open(options.input));
}

void RunTrain(std::ostream& out, const TrainOptions& options) {
    const auto requested = ParseCorpusStorage(options.corpusStorage);
    auto vocabulary = std::make_shared<const Words::Vocabulary>(ReadVocabulary(options.vocabulary));
    auto corpus = std::make_shared<const Words::Corpus>(
        Words::Corpus::Open(options.corpus, requested));

    out << "corpus storage: " << Words::CorpusStorageName(corpus->getStorage()) << '\n';
    if (requested == Words::CorpusStorage::Mapped &&
        corpus->getStorage() == Words::CorpusStorage::Loaded) {
        out << "mmap is not supported here -- fell back to loading the corpus\n";
    }

    Words::Trainer trainer;
    trainer.setVocabulary(vocabulary);
    trainer.setCorpus(corpus);
    trainer.setModelConfig(options.config.model);
    trainer.setSamplingConfig(options.config.sampling);
    trainer.setOutputStream(&out);

    const auto& model = options.config.model;
    const std::size_t rows = 2 * vocabulary->getSize() + model.buckets;

    out << "vocab " << vocabulary->getSize()
        << ", corpus " << corpus->size()
        << ", model "
        << rows * model.dim * sizeof(Words::TFloat) / (1024 * 1024)
        << " MB\n";
    if (model.buckets > 0) {
        out << "subwords: n " << model.minN << ".." << model.maxN
            << ", " << model.buckets << " buckets\n";
    }

    trainer.train(options.config.train);

    Io::WriteFile(options.output,
                  [&](std::ostream& file) {
                      Words::Embeddings::Save(file, trainer.getWordEmbeddings());
                  },
                  std::ios::binary);
    out << "saved embeddings to " << options.output << '\n';

    if (model.buckets > 0) {
        const auto subwordPath = options.subwords.empty()
            ? std::filesystem::path(options.output.string() + ".sub")
            : options.subwords;
        Io::WriteFile(subwordPath,
                      [&](std::ostream& file) {
                          Words::SubwordVectors::Save(
                              file, trainer.getModel()->getSubwordInput(),
                              model.minN, model.maxN, model.buckets);
                      },
                      std::ios::binary);
        out << "saved subword vectors to " << subwordPath << '\n';
    }
}

void RunNeighbours(std::ostream& out, const NeighboursOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);
    RequireMatchingVocabulary(index, vocabulary);

    out << "words " << index.getWords() << ", dim " << index.getDim() << "\n\n";

    if (options.word.empty()) {
        Words::PrintBatteryReport(out, vocabulary, Words::RunDefaultBattery(vocabulary, index));
        return;
    }

    if (!options.subwords.empty() && !vocabulary.getId(options.word).has_value()) {
        const auto subwords = ReadSubwords(options.subwords);
        Words::PrintSubwordNeighbourReport(
            out, vocabulary,
            Words::QuerySubwordNeighbours(index, subwords, options.word, options.count));
        return;
    }

    Words::PrintNeighbourReport(
        out, vocabulary, Words::QueryNeighbours(vocabulary, index, options.word, options.count));
}

void RunEvaluate(std::ostream& out, const EvaluateOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);
    RequireMatchingVocabulary(index, vocabulary);

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
    RequireMatchingVocabulary(index, vocabulary);

    Words::PrintExpressionReport(
        out, vocabulary,
        Words::QueryExpression(vocabulary, index, options.expression, options.count));
}

void RunOddOne(std::ostream& out, const OddOneOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);
    RequireMatchingVocabulary(index, vocabulary);

    Words::PrintOddOneOutReport(
        out, vocabulary, Words::QueryOddOneOut(vocabulary, index, options.words));
}

void RunAxis(std::ostream& out, const AxisOptions& options) {
    const auto vocabulary = ReadVocabulary(options.vocabulary);
    const auto index = ReadEmbeddings(options.embeddings);
    RequireMatchingVocabulary(index, vocabulary);

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
