#include "core/lib/directory_dataset.h"
#include "core/lib/dataset.h"
#include "core/lib/directory_lister.h"
#include <iterator>


namespace Neural {

DirectoryDataset::DirectoryDataset(const std::string& directoryPath)
    : datasetDirectory(directoryPath)
{ }

std::vector<Sample> DirectoryDataset::getSamplesForLabel(const std::size_t label, const std::size_t limit) const
{
    assert(fileReader);
    std::vector<Sample> samples;
    const auto directory = datasetDirectory / std::to_string(label);
    auto files = DirectoryLister::listFiles(directory, limit);
    samples.reserve(files.size());
    std::transform(files.cbegin(), files.cend(), std::back_inserter(samples), [this, label](const auto& file){
        return Sample{
            .input = fileReader(file),
            .label = label,
        };
    });
    return samples;
}

std::vector<Sample> DirectoryDataset::getAllSamples(const std::size_t limitPerLabel) const
{
    assert(fileReader);
    std::vector<Sample> samples;
    const auto directories = DirectoryLister::listDirectories(datasetDirectory);
    for (const auto& directory : directories) {
        const auto label = std::stoull(directory.filename().string());
        auto labelSamples = getSamplesForLabel(label, limitPerLabel);
        samples.insert(
            samples.end(),
            std::make_move_iterator(labelSamples.begin()),
            std::make_move_iterator(labelSamples.end())
        );
    }
    return samples;
}

void DirectoryDataset::setFileReader(const std::function<Matrix(const std::filesystem::path&)>& reader)
{
    fileReader = reader;
}

}
