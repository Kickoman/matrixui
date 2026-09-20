#include <doctest/doctest.h>

#include "core/matrix/matrix.h"
#include "core/nn/neural_network_applier.h"
#include "core/nn/neural_network_loader.h"
#include "core/serving/registry.h"
#include "core/serving/snapshot.h"

#include <atomic>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using Serving::Find;
using Serving::FindDefault;
using Serving::LookupStatus;
using Serving::ModelRegistry;
using Serving::RegistrySnapshot;

// Built in memory: these cases are about the snapshot and the swap, not about
// reading a directory, which model_loader_test already covers.
std::shared_ptr<const Serving::LoadedModel> MakeModel(
    const std::string& name,
    const std::string& version,
    const std::size_t inputSize
) {
    Serving::ModelManifest manifest;
    manifest.manifestVersion = 1;
    manifest.name = name;
    manifest.version = version;
    manifest.input.size = inputSize;
    manifest.output.size = 3;

    Neural::NeuralNetworkConfiguration configuration{};
    configuration.layersSizes = {inputSize, 8, 3};

    return std::make_shared<const Serving::LoadedModel>(
        std::move(manifest),
        std::make_shared<const Neural::NeuralNetwork>(Neural::CreateNetwork(configuration)),
        std::filesystem::path("/models") / (name + "-" + version),
        Serving::IntegrityCheck::NotDeclared);
}

struct Entry {
    std::string name;
    std::string version;
    std::size_t inputSize;
};

std::shared_ptr<RegistrySnapshot> MakeSnapshot(
    const std::vector<Entry>& entries,
    std::map<std::string, std::string> defaults = {}
) {
    auto snapshot = std::make_shared<RegistrySnapshot>();
    for (const auto& entry : entries) {
        snapshot->models.emplace(
            Serving::ModelKey{entry.name, entry.version},
            MakeModel(entry.name, entry.version, entry.inputSize));
    }
    for (auto& [name, version] : defaults) {
        snapshot->defaults.emplace(name, version);
    }
    return snapshot;
}

// What a snapshot holds, as one string. Used to assert that a held composition
// is untouched while the registry swaps underneath it.
std::string Fingerprint(const RegistrySnapshot& snapshot) {
    std::string text;
    for (const auto& [key, model] : snapshot.models) {
        text += key.name + "/" + key.version + ":"
              + std::to_string(model->manifest().input.size) + " ";
    }
    return text;
}

}  // namespace


// The whole point of the step is that what a lookup hands out survives the swap.
// A reference return would let it bind into a temporary; these fail the build.
static_assert(
    std::is_same_v<decltype(std::declval<const ModelRegistry&>().snapshot()),
                   std::shared_ptr<const RegistrySnapshot>>,
    "ModelRegistry::snapshot() must return by value");
static_assert(
    std::is_same_v<decltype(Serving::Lookup{}.model),
                   std::shared_ptr<const Serving::LoadedModel>>,
    "Lookup must own the model it hands out");
static_assert(
    !std::is_copy_constructible_v<ModelRegistry> && !std::is_move_constructible_v<ModelRegistry>,
    "the atomic member makes the registry neither; say so rather than discover it");


TEST_CASE("Find tells the three kinds of miss apart") {
    const auto snapshot = MakeSnapshot({{"mnist", "v1", 64}, {"mnist", "v3", 64}, {"cifar", "v1", 32}});

    const auto hit = Find(*snapshot, "mnist", "v3");
    CHECK(hit.status == LookupStatus::Found);
    REQUIRE(hit.model != nullptr);
    CHECK(hit.model->manifest().version == "v3");

    const auto noModel = Find(*snapshot, "absent", "v1");
    CHECK(noModel.status == LookupStatus::UnknownModel);
    CHECK(noModel.model == nullptr);
    CHECK(noModel.availableVersions.empty());

    const auto noVersion = Find(*snapshot, "mnist", "v9");
    CHECK(noVersion.status == LookupStatus::UnknownVersion);
    CHECK(noVersion.model == nullptr);
    CHECK(noVersion.availableVersions == std::vector<std::string>{"v1", "v3"});
}

TEST_CASE("FindDefault resolves only what the caller declared") {
    const auto withDefault = MakeSnapshot({{"mnist", "v1", 64}, {"mnist", "v3", 64}}, {{"mnist", "v3"}});
    const auto chosen = FindDefault(*withDefault, "mnist");
    CHECK(chosen.status == LookupStatus::Found);
    CHECK(chosen.model->manifest().version == "v3");

    const auto withoutDefault = MakeSnapshot({{"mnist", "v1", 64}, {"mnist", "v3", 64}});
    const auto undecided = FindDefault(*withoutDefault, "mnist");
    CHECK(undecided.status == LookupStatus::NoDefaultVersion);
    CHECK(undecided.availableVersions == std::vector<std::string>{"v1", "v3"});

    // A default naming a version that is not there: the default is broken, the
    // model is not, so explicit versions keep working.
    const auto dangling = MakeSnapshot({{"mnist", "v1", 64}}, {{"mnist", "v9"}});
    CHECK(FindDefault(*dangling, "mnist").status == LookupStatus::UnknownVersion);
    CHECK(Find(*dangling, "mnist", "v1").status == LookupStatus::Found);

    CHECK(FindDefault(*withDefault, "absent").status == LookupStatus::UnknownModel);
}

TEST_CASE("Lookup takes string_view without building a key") {
    // If the comparator stopped being transparent this would not compile.
    static_assert(requires { typename Serving::ModelKeyLess::is_transparent; });

    const auto snapshot = MakeSnapshot({{"mnist", "v1", 64}});
    const std::string name = "mnist";
    const std::string_view view = name;
    CHECK(Find(*snapshot, view, std::string_view{"v1"}).status == LookupStatus::Found);
}

TEST_CASE("A model taken from a snapshot outlives the swap") {
    ModelRegistry registry{Serving::RegistryConfig{}};
    registry.publish(MakeSnapshot({{"mnist", "v3", 64}}));

    const auto held = Find(*registry.snapshot(), "mnist", "v3").model;
    REQUIRE(held != nullptr);

    registry.publish(MakeSnapshot({{"mnist", "v3", 128}}));

    // The request that started on the old composition still sees it.
    CHECK(held->manifest().input.size == 64);
    CHECK(held->network()->inputSize() == 64);
    CHECK(Find(*registry.snapshot(), "mnist", "v3").model->manifest().input.size == 128);
}

TEST_CASE("A snapshot outlives the registry it came from") {
    std::shared_ptr<const RegistrySnapshot> held;
    {
        ModelRegistry registry{Serving::RegistryConfig{}};
        registry.publish(MakeSnapshot({{"mnist", "v3", 64}}));
        held = registry.snapshot();
    }

    const auto found = Find(*held, "mnist", "v3");
    REQUIRE(found.status == LookupStatus::Found);
    const Matrix input(1, 64, 0.5);
    CHECK(Neural::Predict(*found.model->network(), input).getCols() == 3);
}

TEST_CASE("The registry keeps exactly one composition") {
    ModelRegistry registry{Serving::RegistryConfig{}};
    registry.publish(MakeSnapshot({{"mnist", "v1", 64}}));

    std::weak_ptr<const RegistrySnapshot> watch = registry.snapshot();
    CHECK_FALSE(watch.expired());

    registry.publish(MakeSnapshot({{"mnist", "v2", 64}}));

    // Catches a registry that accumulates generations -- the "history of
    // snapshots, for diagnostics" that makes a process grow without bound.
    CHECK(watch.expired());
}

TEST_CASE("publish stamps a strictly increasing generation") {
    ModelRegistry registry{Serving::RegistryConfig{}};
    CHECK(registry.snapshot()->generation == 0);

    registry.publish(MakeSnapshot({{"mnist", "v1", 64}}));
    CHECK(registry.snapshot()->generation == 1);
    registry.publish(MakeSnapshot({{"mnist", "v2", 64}}));
    CHECK(registry.snapshot()->generation == 2);
}

TEST_CASE("Lookups and swaps run together without tearing") {
    // Two readers, not four: the spinlock inside atomic<shared_ptr> is
    // exclusive and spins on pause without yielding, and CI runners have 2-4
    // vCPU. More readers than cores turns this into a thrash that flakes.
    constexpr int kReaders = 2;
    constexpr int kLookups = 50000;
    constexpr int kMinSwaps = 2000;

    ModelRegistry registry{Serving::RegistryConfig{}};

    // The models are built once and shared between compositions: a rebuild does
    // make fresh snapshots, but keeping network construction out of the swap
    // loop puts the contention where the test is looking.
    const auto first = MakeModel("mnist", "v1", 64);
    const auto third = MakeModel("mnist", "v3", 64);
    const auto compose = [&] {
        auto snapshot = std::make_shared<RegistrySnapshot>();
        snapshot->models.emplace(Serving::ModelKey{"mnist", "v1"}, first);
        snapshot->models.emplace(Serving::ModelKey{"mnist", "v3"}, third);
        return snapshot;
    };

    registry.publish(compose());
    const auto held = registry.snapshot();
    const std::string before = Fingerprint(*held);

    std::latch start{kReaders + 2};
    std::atomic<int> inconsistent{0};
    std::atomic<int> missing{0};
    std::atomic<int> readersLeft{kReaders};

    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i) {
        readers.emplace_back([&] {
            int localInconsistent = 0;
            int localMissing = 0;
            start.arrive_and_wait();
            for (int n = 0; n < kLookups; ++n) {
                const auto snapshot = registry.snapshot();
                const auto found = Find(*snapshot, "mnist", "v3");
                if (found.status != LookupStatus::Found) {
                    ++localMissing;
                    continue;
                }
                // An invariant a half-built snapshot would break.
                if (found.model->manifest().input.size != found.model->network()->inputSize()) {
                    ++localInconsistent;
                }
            }
            // No doctest macros on a worker thread; merged after the join.
            inconsistent.fetch_add(localInconsistent, std::memory_order_relaxed);
            missing.fetch_add(localMissing, std::memory_order_relaxed);
            readersLeft.fetch_sub(1, std::memory_order_release);
        });
    }

    std::thread writer([&] {
        start.arrive_and_wait();
        // Bounded by the readers, not by a clock: the swaps have to keep coming
        // for the whole window the readers are in, or the interleaving the test
        // is looking for simply may not happen.
        int swaps = 0;
        while (swaps < kMinSwaps || readersLeft.load(std::memory_order_acquire) > 0) {
            registry.publish(compose());
            ++swaps;
        }
    });

    start.arrive_and_wait();
    for (auto& reader : readers) {
        reader.join();
    }
    writer.join();

    CHECK(inconsistent.load() == 0);
    CHECK(missing.load() == 0);

    // The held composition was never written to, and the registry moved on.
    CHECK(Fingerprint(*held) == before);
    CHECK(held->generation < registry.snapshot()->generation);
}
