#include <doctest/doctest.h>

#include "core/words/expressions.h"
#include "core/words/similarity.h"
#include "core/words/vocabulary.h"
#include "tests/support/fixtures.h"

using namespace Words;

TEST_CASE("ParseExpression resolves a sum of terms with their signs") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    std::string error;
    const auto terms = ParseExpression(vocabulary, "b - a + c", error);

    REQUIRE(terms.size() == 3);
    CHECK(error.empty());
    CHECK(terms[0].id == *vocabulary.getId("b"));
    CHECK(terms[0].sign == doctest::Approx(1.));
    CHECK(terms[1].id == *vocabulary.getId("a"));
    CHECK(terms[1].sign == doctest::Approx(-1.));
    CHECK(terms[2].id == *vocabulary.getId("c"));
    CHECK(terms[2].sign == doctest::Approx(1.));
}

TEST_CASE("ParseExpression handles a single bare word") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    std::string error;
    const auto terms = ParseExpression(vocabulary, "a", error);

    REQUIRE(terms.size() == 1);
    CHECK(terms[0].id == *vocabulary.getId("a"));
    CHECK(terms[0].sign == doctest::Approx(1.));
}

TEST_CASE("ParseExpression reports an unknown word instead of throwing") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    std::string error;
    const auto terms = ParseExpression(vocabulary, "a + zzzznotaword", error);

    CHECK(terms.empty());
    CHECK_FALSE(error.empty());
}

TEST_CASE("ParseExpression reports an empty expression") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    std::string error;
    const auto terms = ParseExpression(vocabulary, "", error);

    CHECK(terms.empty());
    CHECK_FALSE(error.empty());
}

TEST_CASE("BuildExpressionVector applies each term's sign") {
    const Tests::TempDir dir;
    const auto vocabulary = Tests::ToyVocabulary(dir, 1);

    const auto embeddings = Tests::AxisAlignedEmbeddings(6, 8);
    const auto path = dir.file("expr.emb");
    Embeddings::Save(embeddings, path);
    const auto index = EmbeddingIndex::Load(path);

    const std::vector<ExpressionTerm> terms{
        {0, 1.},
        {2, -1.},
    };
    const auto vector = BuildExpressionVector(index, terms);
    REQUIRE(vector.size() == index.getDim());

    const auto* first = index.getNormalized().row(0);
    const auto* second = index.getNormalized().row(2);
    for (std::size_t i = 0; i < index.getDim(); ++i) {
        CHECK(vector[i] == doctest::Approx(first[i] - second[i]).epsilon(1e-6));
    }
}
