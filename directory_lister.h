#ifndef DIRECTORY_LISTER_H
#define DIRECTORY_LISTER_H

#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>

class DirectoryLister {
public:
    static std::vector<std::string> listFiles(const std::string& directoryPath) {
        std::vector<std::string> files;

        // Check if directory exists
        if (!std::filesystem::exists(directoryPath)) {
            throw std::runtime_error("Directory does not exist: " + directoryPath);
        }

        // Check if it's actually a directory
        if (!std::filesystem::is_directory(directoryPath)) {
            throw std::runtime_error("Path is not a directory: " + directoryPath);
        }

        try {
            for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().string();
                    files.push_back(filename);
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Error reading directory: " + std::string(e.what()));
        }

        return files;
    }

    static std::vector<std::string> listFilesWithExtensions(
        const std::string& directoryPath,
        const std::vector<std::string>& extensions,
        const int fileLimit = 0
    ) {
        auto allFiles = listFiles(directoryPath);
        std::vector<std::string> filteredFiles;

        for (const auto& file : allFiles) {
            for (const auto& ext : extensions) {
                if (file.size() >= ext.size() &&
                    file.compare(file.size() - ext.size(), ext.size(), ext) == 0) {
                    filteredFiles.push_back(file);
                    break;
                }
            }
            if (fileLimit > 0 && filteredFiles.size() >= fileLimit) {
                break;
            }
        }

        return filteredFiles;
    }
};

#endif // DIRECTORY_LISTER_H
