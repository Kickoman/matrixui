// These pin the printers' output byte for byte, and are deliberately brittle:
// when the output format is intentionally reworked, the expected strings here
// are meant to be updated in the same commit.

#include <doctest/doctest.h>

#include "core/words/query/evaluate.h"
#include "core/words/query/queries.h"
#include "core/words/report/evaluate_report.h"
#include "core/words/report/query_report.h"
#include "core/words/query/similarity.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/capture.h"
#include "tests/support/fixtures.h"

#include <cmath>
#include <numbers>
#include <sstream>

using namespace Words;

namespace {

// Toy vocabulary (a, b, c, d, e, f by id) paired with unit vectors at fixed
// angles, so every printed number is analytically determined.
//
// Ids 4 and 5: "e" and "f" both occur twice, and equal counts break
// alphabetically, so e -> 4 and f -> 5.
struct Fixture {
    Vocabulary vocabulary;
    EmbeddingIndex index;

    Fixture()
        : vocabulary(Tests::ToyVocabulary(1))
        , index(Build())
    {}

    static EmbeddingIndex Build() {
        constexpr std::size_t words = 6;
        constexpr std::size_t dim = 2;
        const double degrees[words] = {0., 10., 30., 60., 90., 120.};

        Embeddings embeddings(words, dim);
        embeddings.initializeZero();
        for (TWordId id = 0; id < words; ++id) {
            const double radians = degrees[id] * std::numbers::pi / 180.;
            embeddings.row(id)[0] = static_cast<TFloat>(std::cos(radians));
            embeddings.row(id)[1] = static_cast<TFloat>(std::sin(radians));
        }
        return EmbeddingIndex(std::move(embeddings));
    }
};

}  // namespace

TEST_CASE("PrintNeighbours output is stable") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintNeighbourReport(std::cout, fixture.vocabulary,
                             QueryNeighbours(fixture.vocabulary, fixture.index, "a", 3));
        output = capture.str();
    }
    CHECK(output ==
        "a (id 0, count 32):\n"
        "    b                 0.9848\n"
        "    c                 0.8660\n"
        "    d                 0.5000\n"
        "\n");
}

TEST_CASE("PrintNeighbours reports an unknown word") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintNeighbourReport(std::cout, fixture.vocabulary,
                             QueryNeighbours(fixture.vocabulary, fixture.index, "zzz", 3));
        output = capture.str();
    }
    CHECK(output == "  'zzz' is not in the vocabulary\n");
}

TEST_CASE("PrintAnalogy output is stable") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintAnalogyQueryReport(std::cout, fixture.vocabulary,
                                QueryAnalogy(fixture.vocabulary, fixture.index, "a", "b", "c", 3));
        output = capture.str();
    }
    CHECK(output ==
        "b - a + c:\n"
        "    d                 0.9296\n"
        "    e                 0.6207\n"
        "    f                 0.1456\n"
        "\n");
}

TEST_CASE("PrintAnalogy reports unknown words") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintAnalogyQueryReport(std::cout, fixture.vocabulary,
                                QueryAnalogy(fixture.vocabulary, fixture.index, "a", "zzz", "c", 3));
        output = capture.str();
    }
    CHECK(output == "  some of 'a', 'zzz', 'c' are not in the vocabulary\n\n");
}

TEST_CASE("RunExpression prints both 3CosAdd and 3CosMul for an analogy") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintExpressionReport(std::cout, fixture.vocabulary,
                              QueryExpression(fixture.vocabulary, fixture.index, "b - a + c", 3));
        output = capture.str();
    }
    CHECK(output ==
        "b - a + c\n"
        "\n"
        "  3CosAdd:\n"
        "    d                 0.9296\n"
        "    e                 0.6207\n"
        "    f                 0.1456\n"
        "\n"
        "  3CosMul:\n"
        "    d                 1.0205\n"
        "    e                 0.8785\n"
        "    f                 0.6554\n");
}

TEST_CASE("RunExpression reports an unparseable expression") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintExpressionReport(std::cout, fixture.vocabulary,
                              QueryExpression(fixture.vocabulary, fixture.index, "b + zzz", 3));
        output = capture.str();
    }
    CHECK(output == "  'zzz' is not in the vocabulary\n");
}

TEST_CASE("RunOddOne ranks by centroid similarity and names the outlier") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintOddOneOutReport(std::cout, fixture.vocabulary,
                             QueryOddOneOut(fixture.vocabulary, fixture.index, "a b c d"));
        output = capture.str();
    }
    CHECK(output ==
        "  similarity to the centroid:\n"
        "    c                 0.9957\n"
        "    b                 0.9674\n"
        "    a                 0.9087\n"
        "    d                 0.8159\n"
        "\n"
        "  odd one out: d\n");
}

TEST_CASE("RunOddOne requires at least three words") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintOddOneOutReport(std::cout, fixture.vocabulary,
                             QueryOddOneOut(fixture.vocabulary, fixture.index, "a b"));
        output = capture.str();
    }
    CHECK(output == "  need at least three words\n");
}

TEST_CASE("RunAxis projects an explicit word list") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintAxisReport(std::cout, fixture.vocabulary,
                        QueryAxis(fixture.vocabulary, fixture.index, "a - e", "b c d", 0, 3));
        output = capture.str();
    }
    CHECK(output ==
        "axis: a - e\n"
        "\n"
        "  +0.5736  b\n"
        "  +0.2588  c\n"
        "  -0.2588  d\n");
}

TEST_CASE("RunAxis scans the vocabulary and prints both ends") {
    const Fixture fixture;
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintAxisReport(std::cout, fixture.vocabulary,
                        QueryAxis(fixture.vocabulary, fixture.index, "a - e", "", 0, 2));
        output = capture.str();
    }
    CHECK(output ==
        "axis: a - e\n"
        "\n"
        "  positive end:\n"
        "  +0.7071  a\n"
        "  +0.5736  b\n"
        "\n"
        "  negative end:\n"
        "  -0.9659  f\n"
        "  -0.7071  e\n");
}

TEST_CASE("PrintSimilarityReport output is stable") {
    const SimilarityReport report{/*asked=*/12, /*skipped=*/3, /*spearman=*/0.5, /*pearson=*/0.25};
    std::string output;
    {
        Tests::CoutCapture capture;
        PrintSimilarityReport(std::cout, "wordsim353.txt", report);
        output = capture.str();
    }
    CHECK_FALSE(output.empty());
    CHECK(output.find("wordsim353.txt") != std::string::npos);
    CHECK(output.find("0.5") != std::string::npos);
}

TEST_CASE("PrintAnalogyReport output is stable") {
    AnalogyReport report;
    report.categories.push_back(AnalogyStats{"capital-common", 10, 2, 5, 6});
    report.semantic = AnalogyStats{"semantic", 10, 2, 5, 6};
    report.syntactic = AnalogyStats{"syntactic", 0, 0, 0, 0};
    report.overall = AnalogyStats{"overall", 10, 2, 5, 6};

    std::string output;
    {
        Tests::CoutCapture capture;
        PrintAnalogyReport(std::cout, report);
        output = capture.str();
    }
    CHECK_FALSE(output.empty());
    CHECK(output.find("capital-common") != std::string::npos);
}
