#include "core/words/query/evaluate.h"

#include "core/lib/stats.h"
#include "core/lib/text.h"
#include "core/words/data/vocabulary.h"

#include <algorithm>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace Words {

namespace {

struct Question {
    TWordId a;
    TWordId b;
    TWordId c;
    TWordId expected;
    std::size_t category;
};

bool IsSemantic(const std::string& name) {
    return name.rfind("gram", 0) != 0;
}

void Accumulate(AnalogyStats& target, const AnalogyStats& source) {
    target.asked += source.asked;
    target.skipped += source.skipped;
    target.correctAdd += source.correctAdd;
    target.correctMul += source.correctMul;
}

void AnswerQuestion(
    const EmbeddingIndex& index,
    const Question& question,
    const std::size_t limit,
    TWordId& bestAdd,
    TWordId& bestMul
) {
    const auto dim = index.getDim();
    const auto& embeddings = index.getNormalized();

    const auto* vectorA = embeddings.row(question.a);
    const auto* vectorB = embeddings.row(question.b);
    const auto* vectorC = embeddings.row(question.c);

    double topAdd = -std::numeric_limits<double>::infinity();
    double topMul = -std::numeric_limits<double>::infinity();
    bestAdd = 0;
    bestMul = 0;

    for (TWordId id = 0; id < limit; ++id) {
        if (id == question.a || id == question.b || id == question.c) {
            continue;
        }
        const auto* candidate = embeddings.row(id);

        const double simA = dot(candidate, vectorA, dim);
        const double simB = dot(candidate, vectorB, dim);
        const double simC = dot(candidate, vectorC, dim);

        const double scoreAdd = simB - simA + simC;
        if (scoreAdd > topAdd) {
            topAdd = scoreAdd;
            bestAdd = id;
        }

        const double shiftedA = (simA + 1.) / 2.;
        const double shiftedB = (simB + 1.) / 2.;
        const double shiftedC = (simC + 1.) / 2.;
        const double scoreMul = shiftedB * shiftedC / (shiftedA + 0.001);
        if (scoreMul > topMul) {
            topMul = scoreMul;
            bestMul = id;
        }
    }
}

}  // namespace

AnalogyReport EvaluateAnalogies(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    std::istream& file,
    const std::size_t restrictTo,
    const std::size_t threads
) {
    AnalogyReport report;
    std::vector<Question> questions;

    std::string line;
    std::size_t currentCategory = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == ':') {
            std::string name = line.substr(1);
            name.erase(0, name.find_first_not_of(" \t"));
            report.categories.push_back(AnalogyStats{name, 0, 0, 0, 0});
            currentCategory = report.categories.size() - 1;
            continue;
        }
        if (report.categories.empty()) {
            continue;
        }

        std::istringstream stream(line);
        std::string wordA;
        std::string wordB;
        std::string wordC;
        std::string wordD;
        if (!(stream >> wordA >> wordB >> wordC >> wordD)) {
            continue;
        }

        const auto idA = vocabulary.getId(ToLower(wordA));
        const auto idB = vocabulary.getId(ToLower(wordB));
        const auto idC = vocabulary.getId(ToLower(wordC));
        const auto idD = vocabulary.getId(ToLower(wordD));

        if (!idA.has_value() || !idB.has_value() || !idC.has_value() || !idD.has_value()) {
            ++report.categories[currentCategory].skipped;
            continue;
        }

        questions.push_back(Question{*idA, *idB, *idC, *idD, currentCategory});
    }

    const std::size_t limit = restrictTo > 0
        ? std::min<std::size_t>(restrictTo, index.getWords())
        : index.getWords();

    const std::size_t threadCount = threads > 0
        ? threads
        : std::max<std::size_t>(1, std::thread::hardware_concurrency());

    std::vector<std::vector<AnalogyStats>> perThread(
        threadCount, std::vector<AnalogyStats>(report.categories.size()));

    std::vector<std::thread> workers;
    workers.reserve(threadCount);

    const std::size_t perWorker = questions.size() / threadCount + 1;

    for (std::size_t worker = 0; worker < threadCount; ++worker) {
        workers.emplace_back([&, worker] {
            const std::size_t from = worker * perWorker;
            const std::size_t to = std::min(from + perWorker, questions.size());

            for (std::size_t i = from; i < to; ++i) {
                const auto& question = questions[i];

                TWordId bestAdd = 0;
                TWordId bestMul = 0;
                AnswerQuestion(index, question, limit, bestAdd, bestMul);

                auto& stats = perThread[worker][question.category];
                ++stats.asked;
                if (bestAdd == question.expected) {
                    ++stats.correctAdd;
                }
                if (bestMul == question.expected) {
                    ++stats.correctMul;
                }
            }
        });
    }

    for (auto& thread : workers) {
        thread.join();
    }

    for (const auto& threadStats : perThread) {
        for (std::size_t category = 0; category < report.categories.size(); ++category) {
            Accumulate(report.categories[category], threadStats[category]);
        }
    }

    report.semantic.name = "semantic";
    report.syntactic.name = "syntactic";
    report.overall.name = "total";

    for (const auto& category : report.categories) {
        Accumulate(IsSemantic(category.name) ? report.semantic : report.syntactic, category);
        Accumulate(report.overall, category);
    }

    return report;
}

SimilarityReport EvaluateSimilarity(
    const Vocabulary& vocabulary,
    const EmbeddingIndex& index,
    std::istream& file,
    const std::size_t scoreColumn
) {
    SimilarityReport report;
    std::vector<double> human;
    std::vector<double> model;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields;
        std::string field;
        std::istringstream stream(line);
        while (stream >> field) {
            fields.push_back(field);
        }

        if (fields.size() <= scoreColumn) {
            continue;
        }

        double score = 0.;
        try {
            score = std::stod(fields[scoreColumn]);
        } catch (const std::exception&) {
            continue;
        }

        const auto first = vocabulary.getId(ToLower(fields[0]));
        const auto second = vocabulary.getId(ToLower(fields[1]));

        if (!first.has_value() || !second.has_value()) {
            ++report.skipped;
            continue;
        }

        human.push_back(score);
        model.push_back(index.similarity(*first, *second));
        ++report.asked;
    }

    report.pearson = PearsonOf(human, model);
    report.spearman = PearsonOf(RanksOf(human), RanksOf(model));
    return report;
}

}
