#include "core/words_cli/commands.h"

#include "core/words/corpus.h"
#include "core/words/embeddings.h"
#include "core/words/evaluate.h"
#include "core/words/queries.h"
#include "core/words/report/evaluate_report.h"
#include "core/words/report/inspect_report.h"
#include "core/words/report/query_report.h"
#include "core/words/similarity.h"
#include "core/words/utils/inspect.h"
#include "core/words/trainer.h"
#include "core/words/vocabulary.h"

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

}  // namespace

void Inspect(std::ostream& out, const InspectOptions& options) {
    Words::PrintCorpusStatistics(out, Words::InspectDump(options.input, options.topN));
}

void BuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options) {
    const auto vocabulary = Words::Vocabulary::Build(options.input, options.minCount);
    PrintVocabularyInfo(out, vocabulary);
    Words::Vocabulary::Save(vocabulary, options.output);
}

void LoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options) {
    PrintVocabularyInfo(out, Words::Vocabulary::Load(options.input));
}

void BuildCorpus(std::ostream& out, const BuildCorpusOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto corpus = Words::EncodeCorpus(options.input, vocabulary);
    PrintCorpusInfo(out, corpus);
    Words::SaveCorpus(options.output, corpus);
}

void LoadCorpus(std::ostream& out, const LoadCorpusOptions& options) {
    PrintCorpusInfo(out, Words::LoadCorpus(options.input));
}

void Train(std::ostream& out, const TrainOptions& options) {
    auto vocabulary =
        std::make_shared<const Words::Vocabulary>(Words::Vocabulary::Load(options.vocabulary));
    auto corpus = std::make_shared<const Words::TCorpus>(Words::LoadCorpus(options.corpus));

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

    Words::Embeddings::Save(trainer.getInputEmbeddings(), options.output);
    out << "saved embeddings to " << options.output << '\n';
}

void Neighbours(std::ostream& out, const NeighboursOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    out << "words " << index.getWords() << ", dim " << index.getDim() << "\n\n";

    if (options.word.empty()) {
        Words::PrintBatteryReport(out, vocabulary, Words::RunDefaultBattery(vocabulary, index));
    } else {
        Words::PrintNeighbourReport(
            out, vocabulary, Words::QueryNeighbours(vocabulary, index, options.word, options.count));
    }
}

void Evaluate(std::ostream& out, const EvaluateOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    out << "words " << index.getWords() << ", dim " << index.getDim() << '\n';

    if (!options.analogies.empty()) {
        out << "\n=== Analogies ===\n";
        Words::PrintAnalogyReport(
            out,
            Words::EvaluateAnalogies(
                vocabulary, index, options.analogies, options.restrictTo, options.threads));
    }

    if (!options.similarity.empty()) {
        out << "\n=== Similarity ===\n";
        Words::PrintSimilarityReport(
            out,
            options.similarity.filename().string(),
            Words::EvaluateSimilarity(vocabulary, index, options.similarity, options.scoreColumn));
    }
}

void Expression(std::ostream& out, const ExpressionOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintExpressionReport(
        out, vocabulary,
        Words::QueryExpression(vocabulary, index, options.expression, options.count));
}

void OddOne(std::ostream& out, const OddOneOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintOddOneOutReport(
        out, vocabulary, Words::QueryOddOneOut(vocabulary, index, options.words));
}

void Axis(std::ostream& out, const AxisOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintAxisReport(
        out, vocabulary,
        Words::QueryAxis(vocabulary, index, options.axis, options.words,
                         options.restrictTo, options.count));
}

}  // namespace WordsCli
