#include "core/words/config.h"

#include "core/words/error.h"

#include <cstdint>
#include <limits>
#include <string>

namespace Words {

void Validate(const WordsConfig& config, const std::size_t vocabularySize) {
    if (vocabularySize == 0) {
        throw ConfigError("vocabulary is empty");
    }
    if (config.model.dim == 0) {
        throw ConfigError("dim must be greater than zero");
    }
    if (config.model.negatives == 0) {
        throw ConfigError("negatives must be greater than zero");
    }
    if (config.model.initialLearningRate <= 0.) {
        throw ConfigError("learning rate must be positive");
    }
    if (config.model.buckets > 0) {
        if (config.model.minN == 0) {
            throw ConfigError("min-n must be greater than zero when subwords are enabled");
        }
        if (config.model.maxN < config.model.minN) {
            throw ConfigError(
                "max-n (" + std::to_string(config.model.maxN) + ") is smaller than min-n ("
                + std::to_string(config.model.minN) + ")");
        }
        if (config.model.buckets > std::numeric_limits<std::uint32_t>::max()) {
            throw ConfigError("buckets must fit into 32 bits");
        }
    }
    if (config.sampling.window == 0) {
        throw ConfigError("window must be greater than zero");
    }
    if (config.sampling.sample < 0.) {
        throw ConfigError("sample threshold cannot be negative");
    }
    if (config.sampling.negativeTableSize < vocabularySize) {
        throw ConfigError(
            "negative sampling table (" + std::to_string(config.sampling.negativeTableSize)
            + ") is smaller than the vocabulary (" + std::to_string(vocabularySize) + ")");
    }
    if (config.train.epochs == 0) {
        throw ConfigError("epochs must be greater than zero");
    }
    if (config.train.chunkSize == 0) {
        throw ConfigError("chunk size must be greater than zero");
    }
}

}  // namespace Words
