#pragma once

#include "core/generator/learning_config.h"

#include <cstddef>
#include <string>
#include <vector>

namespace GeneratorCli {

struct GenerateOptions {
    std::size_t label{0};
    std::size_t numClasses{10};
    std::size_t numSamples{1};
    std::string generatorPath{"generator.wgt"};
    std::string outputPath{"generated.png"};
    std::string classifierPath;
    std::size_t imageWidth{28};
    std::size_t imageHeight{28};
};

struct TrainOptions {
    std::string classifierPath;
    std::string datasetPath;
    std::string generatorPath{"generator.wgt"};
    std::string discriminatorPath{"discriminator.wgt"};
    std::vector<std::size_t> generatorHidden{256, 512};
    std::vector<std::size_t> discriminatorHidden{512, 256};
    std::size_t imageWidth{28};
    std::size_t imageHeight{28};
    Neural::GAN::LearningConfig config{};
};

}  // namespace GeneratorCli
