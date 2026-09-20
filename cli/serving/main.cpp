#include "cli/serving/commands.h"
#include "cli/serving/options.h"

#include <CLI11/CLI11.hpp>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"Inspect a directory of model artifacts"};
    app.require_subcommand(1);

    int exitCode = ServingCli::kSuccess;

    ServingCli::ListOptions list;
    std::vector<std::string> defaults;

    auto* listCmd = app.add_subcommand("list", "Load every model under a root and print what came up");
    listCmd->add_option("--root", list.root, "Directory holding one subdirectory per model")
        ->required()
        ->check(CLI::ExistingDirectory);
    listCmd->add_option("--default", defaults,
            "Version a request without one resolves to, as name=version. Repeatable.")
        ->take_all();

    listCmd->callback([&] {
        for (const auto& entry : defaults) {
            const auto split = entry.find('=');
            if (split == std::string::npos || split == 0 || split + 1 == entry.size()) {
                std::cerr << "--default expects name=version, got: " << entry << "\n";
                exitCode = ServingCli::kBadOption;
                return;
            }
            list.defaults.emplace(entry.substr(0, split), entry.substr(split + 1));
        }
        exitCode = ServingCli::List(std::cout, std::cerr, list);
    });

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    return exitCode;
}
