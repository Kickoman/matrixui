#include "core/nn/dataset.h"
#include <filesystem>
#include <functional>


namespace Neural {


class DirectoryDataset : public Dataset {
public:
    DirectoryDataset(const std::string& directoryPath);

    static bool IsDirectoryValid(const std::filesystem::path& path, std::size_t classCount);

    std::vector<Sample> getSamplesForLabel(const std::size_t label, const std::size_t limit = 0) const override;
    std::vector<Sample> getAllSamples(std::size_t limitPerLabel = 0) const override;

    void setFileReader(const std::function<Matrix(const std::filesystem::path&)>& reader);

private:
    std::filesystem::path datasetDirectory;
    std::function<Matrix(const std::filesystem::path&)> fileReader;
};


}
