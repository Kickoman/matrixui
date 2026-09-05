#include <doctest/doctest.h>

#include "core/words/report/inspect_report.h"
#include "core/words/data/inspect.h"
#include "tests/support/fixtures.h"

#include <sstream>

using namespace Words;

TEST_CASE("InspectDump counts tokens, symbols and distinct words") {
    const auto statistics = Tests::InspectText(Tests::ToyCorpusText());

    CHECK(statistics.totalWords == 64);
    CHECK(statistics.uniqueWords == 6);
    CHECK(statistics.symbols == 64);  // every fixture word is one character
}

TEST_CASE("InspectDump reports how many words survive each min-count") {
    const auto statistics = Tests::InspectText(Tests::ToyCorpusText());
    REQUIRE(statistics.survivorsByMinCount.size() == 3);

    // Counts are a=32, b=16, c=8, d=4, e=2, f=2.
    CHECK(statistics.survivorsByMinCount[0].first == 5);
    CHECK(statistics.survivorsByMinCount[0].second == 3);   // a, b, c
    CHECK(statistics.survivorsByMinCount[1].first == 10);
    CHECK(statistics.survivorsByMinCount[1].second == 2);   // a, b
    CHECK(statistics.survivorsByMinCount[2].first == 50);
    CHECK(statistics.survivorsByMinCount[2].second == 0);
}

TEST_CASE("InspectDump ranks the most frequent words with their share") {
    const auto statistics = Tests::InspectText(Tests::ToyCorpusText());
    REQUIRE(statistics.topByFrequency.size() == 6);

    CHECK(statistics.topByFrequency[0].word == "a");
    CHECK(statistics.topByFrequency[0].count == 32);
    CHECK(statistics.topByFrequency[0].share == doctest::Approx(0.5));

    for (std::size_t i = 1; i < statistics.topByFrequency.size(); ++i) {
        CHECK(statistics.topByFrequency[i].count <= statistics.topByFrequency[i - 1].count);
    }
}

TEST_CASE("InspectDump honours the topN limit") {
    CHECK(Tests::InspectText(Tests::ToyCorpusText(), 2).topByFrequency.size() == 2);
}

TEST_CASE("PrintCorpusStatistics writes to the stream it is given") {
    std::ostringstream out;
    PrintCorpusStatistics(out, "dump.txt", Tests::InspectText(Tests::ToyCorpusText()));

    CHECK(out.str().find("Words: 64") != std::string::npos);
    CHECK(out.str().find("Unique words: 6") != std::string::npos);
    CHECK(out.str().find("minCount = 5") != std::string::npos);
}
