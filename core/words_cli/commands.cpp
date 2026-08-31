#include "core/words_cli/commands.h"

#include "core/words/corpus.h"
#include "core/words/diagnostics/diagnostics.h"
#include "core/words/embeddings.h"
#include "core/words/evaluate.h"
#include "core/words/queries.h"
#include "core/words/report/diagnostics_report.h"
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

bool Inspect(std::ostream& out, const InspectOptions& options) {
    Words::PrintCorpusStatistics(out, Words::InspectDump(options.input, options.topN));
    return true;
}

bool BuildVocabulary(std::ostream& out, const BuildVocabularyOptions& options) {
    const auto vocabulary = Words::Vocabulary::Build(options.input, options.minCount);
    PrintVocabularyInfo(out, vocabulary);
    Words::Vocabulary::Save(vocabulary, options.output);
    return true;
}

bool LoadVocabulary(std::ostream& out, const LoadVocabularyOptions& options) {
    PrintVocabularyInfo(out, Words::Vocabulary::Load(options.input));
    return true;
}

bool BuildCorpus(std::ostream& out, const BuildCorpusOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto corpus = Words::EncodeCorpus(options.input, vocabulary);
    PrintCorpusInfo(out, corpus);
    Words::SaveCorpus(options.output, corpus);
    return true;
}

bool LoadCorpus(std::ostream& out, const LoadCorpusOptions& options) {
    PrintCorpusInfo(out, Words::LoadCorpus(options.input));
    return true;
}

bool ValidateSubsampler(std::ostream& out, const ValidateSubsamplerOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto corpus = Words::LoadCorpus(options.corpus);

    const auto report = Words::Diagnostics::CheckSubsampler(vocabulary, corpus, options.sample);
    Words::Diagnostics::PrintSubsamplerReport(out, vocabulary, report);
    return report.allPassed();
}

bool ValidateWindowSampler(std::ostream& out, const ValidateWindowSamplerOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto corpus = Words::LoadCorpus(options.corpus);

    const auto report = Words::Diagnostics::CheckWindowSampler(vocabulary, corpus);
    Words::Diagnostics::PrintWindowSamplerReport(out, vocabulary, report);
    return report.allPassed();
}

bool ValidateNegativeSampler(std::ostream& out, const ValidateVocabularyOnlyOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);

    const auto report = Words::Diagnostics::CheckNegativeSampler(vocabulary);
    Words::Diagnostics::PrintNegativeSamplerReport(out, vocabulary, report);
    return report.allPassed();
}

bool ValidateModel(std::ostream& out, const ValidateVocabularyOnlyOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);

    Words::ModelConfig config;
    config.dim = 100;

    const auto report = Words::Diagnostics::CheckModelInit(vocabulary, config);
    Words::Diagnostics::PrintModelInitReport(out, report);
    return report.allPassed();
}

bool ValidateGradients(std::ostream& out, const ValidateVocabularyOnlyOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);

    Words::ModelConfig config;
    config.dim = 100;

    const auto init = Words::Diagnostics::CheckModelInit(vocabulary, config);
    Words::Diagnostics::PrintModelInitReport(out, init);
    out << '\n';

    const auto gradients = Words::Diagnostics::CheckGradients(vocabulary);
    Words::Diagnostics::PrintGradientReport(out, gradients);

    const auto loss = Words::Diagnostics::CheckLossBehaviour(vocabulary);
    Words::Diagnostics::PrintLossBehaviourReport(out, loss);

    return init.allPassed() && gradients.allPassed() && loss.allPassed();
}

bool Train(std::ostream& out, const TrainOptions& options) {
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
    return true;
}

bool Neighbours(std::ostream& out, const NeighboursOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    out << "words " << index.getWords() << ", dim " << index.getDim() << "\n\n";

    if (options.word.empty()) {
        Words::PrintBatteryReport(out, vocabulary, Words::RunDefaultBattery(vocabulary, index));
    } else {
        Words::PrintNeighbourReport(
            out, vocabulary, Words::QueryNeighbours(vocabulary, index, options.word, options.count));
    }
    return true;
}

bool Evaluate(std::ostream& out, const EvaluateOptions& options) {
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
    return true;
}

bool Expression(std::ostream& out, const ExpressionOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintExpressionReport(
        out, vocabulary,
        Words::QueryExpression(vocabulary, index, options.expression, options.count));
    return true;
}

bool OddOne(std::ostream& out, const OddOneOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintOddOneOutReport(
        out, vocabulary, Words::QueryOddOneOut(vocabulary, index, options.words));
    return true;
}

bool Axis(std::ostream& out, const AxisOptions& options) {
    const auto vocabulary = Words::Vocabulary::Load(options.vocabulary);
    const auto index = Words::EmbeddingIndex::Load(options.embeddings);

    Words::PrintAxisReport(
        out, vocabulary,
        Words::QueryAxis(vocabulary, index, options.axis, options.words,
                         options.restrictTo, options.count));
    return true;
}

}  // namespace WordsCli
