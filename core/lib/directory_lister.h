#ifndef DIRECTORY_LISTER_H
#define DIRECTORY_LISTER_H

#include <algorithm>
#include <cstddef>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>

class DirectoryLister {
public:
    // Entries whose name starts with a dot are never data. Skipping them is what
    // makes a dataset directory usable on macOS, where Finder drops .DS_Store into
    // every folder it displays and a volume root carries .Spotlight-V100 and
    // .fseventsd. On Linux this only ever skips things that were not ours anyway.
    static bool isHidden(const std::filesystem::path& path) {
        const auto name = path.filename().string();
        return !name.empty() && name.front() == '.';
    }

    static std::vector<std::filesystem::path> listDirectories(const std::string& directoryPath) {
        std::vector<std::filesystem::path> directories;
        if (!std::filesystem::exists(directoryPath)) {
            throw std::runtime_error("Directory does not exist: " + directoryPath);
        }
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.is_directory() && !isHidden(entry.path())) {
                directories.push_back(entry.path());
            }
        }
        // directory_iterator yields in an unspecified order that differs between
        // filesystems (ext4 vs APFS), so sort to keep runs reproducible.
        std::sort(directories.begin(), directories.end());
        return directories;
    }

    static std::vector<std::string> listFiles(const std::string& directoryPath, const std::size_t limit = 0) {
        return collect(directoryPath, {}, limit);
    }

    static std::vector<std::string> listFilesWithExtensions(
        const std::string& directoryPath,
        const std::vector<std::string>& extensions,
        const std::size_t fileLimit = 0
    ) {
        return collect(directoryPath, extensions, fileLimit);
    }

private:
    // Collects every visible regular file, optionally filtered by extension, then
    // sorts before applying the limit. Truncating an unordered listing would make
    // `limit` select a different subset per platform and silently change results.
    static std::vector<std::string> collect(
        const std::string& directoryPath,
        const std::vector<std::string>& extensions,
        const std::size_t limit
    ) {
        if (!std::filesystem::exists(directoryPath)) {
            throw std::runtime_error("Directory does not exist: " + directoryPath);
        }

        if (!std::filesystem::is_directory(directoryPath)) {
            throw std::runtime_error("Path is not a directory: " + directoryPath);
        }

        std::vector<std::string> files;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
                if (!entry.is_regular_file() || isHidden(entry.path())) {
                    continue;
                }
                const auto file = entry.path().string();
                if (extensions.empty() || hasAnyExtension(file, extensions)) {
                    files.push_back(file);
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            throw std::runtime_error("Error reading directory: " + std::string(e.what()));
        }

        std::sort(files.begin(), files.end());
        if (limit > 0 && files.size() > limit) {
            files.resize(limit);
        }
        return files;
    }

    static bool hasAnyExtension(const std::string& file, const std::vector<std::string>& extensions) {
        for (const auto& ext : extensions) {
            if (file.size() >= ext.size() &&
                file.compare(file.size() - ext.size(), ext.size(), ext) == 0) {
                return true;
            }
        }
        return false;
    }
};

#endif // DIRECTORY_LISTER_H
