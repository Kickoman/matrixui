#include <doctest/doctest.h>

#include "core/words/query/similarity.h"
#include "core/words/data/vocabulary.h"
#include "tests/support/fixtures.h"
#include "tests/support/temp_dir.h"

#include <cmath>

using namespace Words;

namespace {

// EmbeddingIndex has no public constructor, so a toy index has to go through
// the filesystem. Keep words <= dim so no two rows collapse onto the same axis.
EmbeddingIndex ToyIndex(const Tests::TempDir& dir, const std::size_t words = 6, const std::size_t dim = 8) {
    const auto embeddings = Tests::AxisAlignedEmbeddings(words, dim);
    const auto path = dir.file("toy.emb");
    Embeddings::Save(embeddings, path);
    return EmbeddingIndex::Load(path);
}

}  // namespace

TEST_CASE("The index reports the shape it was loaded with") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir, 6, 8);
    CHECK(index.getWords() == 6);
    CHECK(index.getDim() == 8);
}

TEST_CASE("Loading normalises every row to unit length") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    for (TWordId id = 0; id < index.getWords(); ++id) {
        CHECK(index.similarity(id, id) == doctest::Approx(1.).epsilon(1e-6));
    }
}

TEST_CASE("Similarity is symmetric and matches the hand-computed cosine") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    CHECK(index.similarity(0, 1) == doctest::Approx(index.similarity(1, 0)));

    // Row 0 is (1,0,...) and row 1 is (2,1,0,...), so cos = 2/sqrt(5).
    CHECK(index.similarity(0, 1) == doctest::Approx(2. / std::sqrt(5.)).epsilon(1e-6));
    // Rows 2 and 3 sit on orthogonal axes.
    CHECK(index.similarity(2, 3) == doctest::Approx(0.).epsilon(1e-6));
}

TEST_CASE("nearest returns the closest row and never the query itself") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    const auto neighbours = index.nearest(0, 3);
    REQUIRE(neighbours.size() == 3);

    // Row 1 is the only row not orthogonal to row 0.
    CHECK(neighbours[0].id == 1);
    CHECK(neighbours[0].similarity == doctest::Approx(2. / std::sqrt(5.)).epsilon(1e-6));

    for (const auto& neighbour : neighbours) {
        CHECK(neighbour.id != 0);
    }
}

TEST_CASE("nearest returns results in descending similarity") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    const auto neighbours = index.nearest(1, 5);
    REQUIRE(neighbours.size() == 5);
    for (std::size_t i = 1; i < neighbours.size(); ++i) {
        CHECK(neighbours[i].similarity <= neighbours[i - 1].similarity);
    }
}

TEST_CASE("nearest caps the result at the number of candidates") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir, 4, 8);
    // 4 words, one excluded as the query itself.
    CHECK(index.nearest(0, 100).size() == 3);
}

TEST_CASE("nearestToVector honours the exclusion list") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    const std::vector<TFloat> query(index.getNormalized().row(0),
                                    index.getNormalized().row(0) + index.getDim());
    const std::array<TWordId, 2> exclude{0, 1};

    for (const auto& neighbour : index.nearestToVector(query, exclude, 4)) {
        CHECK(neighbour.id != 0);
        CHECK(neighbour.id != 1);
    }
}

TEST_CASE("analogyVector computes b - a + c") {
    const Tests::TempDir dir;
    const auto index = ToyIndex(dir);

    const auto result = index.analogyVector(0, 1, 2);
    REQUIRE(result.size() == index.getDim());

    const auto* a = index.getNormalized().row(0);
    const auto* b = index.getNormalized().row(1);
    const auto* c = index.getNormalized().row(2);
    for (std::size_t i = 0; i < index.getDim(); ++i) {
        CHECK(result[i] == doctest::Approx(b[i] - a[i] + c[i]).epsilon(1e-6));
    }
}
