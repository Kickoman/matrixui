#include "core/words/config.h"

#include "core/words/error.h"

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
