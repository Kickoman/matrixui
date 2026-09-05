#include "core/nn/directory_dataset.h"
#include "core/nn/dataset.h"
#include "core/lib/directory_lister.h"
#include <iterator>


namespace Neural {

bool DirectoryDataset::IsDirectoryValid(const std::filesystem::path& path, const std::size_t classCount) {
    for (std::size_t i = 0; i < classCount; ++i) {
        const auto subdirectory = path / std::to_string(i);
        if (!std::filesystem::exists(subdirectory) || !std::filesystem::is_directory(subdirectory)) {
            return false;
        }
    }
    return true;
}

DirectoryDataset::DirectoryDataset(const std::string& directoryPath)
    : datasetDirectory(directoryPath)
{ }

std::vector<Sample> DirectoryDataset::getSamplesForLabel(const std::size_t label, const std::size_t limit) const
{
    assert(fileReader);
    std::vector<Sample> samples;
    const auto directory = datasetDirectory / std::to_string(label);
    auto files = DirectoryLister::listFilesWithExtensions(directory, {".png", ".PNG"}, limit);
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
        // A dataset root can hold directories that are not class labels -- on macOS
        // a volume carries .Spotlight-V100 and .fseventsd. std::stoull would throw
        // on those, so skip anything that is not purely a number instead.
        const auto name = directory.filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        const auto label = std::stoull(name);
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
