#include "cli/serving/commands.h"
#include "cli/serving/options.h"

#include <CLI11/CLI11.hpp>

#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CLI::App app{"Inspect a directory of model artifacts"};
    app.require_subcommand(1);

    int exitCode = ServingCli::kSuccess;

    const auto parseDefaults = [&](const std::vector<std::string>& entries,
                                   std::map<std::string, std::string>& into) {
        for (const auto& entry : entries) {
            const auto split = entry.find('=');
            if (split == std::string::npos || split == 0 || split + 1 == entry.size()) {
                std::cerr << "--default expects name=version, got: " << entry << "\n";
                return false;
            }
            into.emplace(entry.substr(0, split), entry.substr(split + 1));
        }
        return true;
    };

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
        if (!parseDefaults(defaults, list.defaults)) {
            exitCode = ServingCli::kBadOption;
            return;
        }
        exitCode = ServingCli::List(std::cout, std::cerr, list);
    });

    ServingCli::ServeOptions serve;
    std::vector<std::string> serveDefaults;

    auto* serveCmd = app.add_subcommand("serve", "Serve the models under a root over HTTP");
    serveCmd->add_option("--root", serve.root, "Directory holding one subdirectory per model")
        ->required()
        ->check(CLI::ExistingDirectory);
    serveCmd->add_option("--default", serveDefaults,
            "Version a request without one resolves to, as name=version. Repeatable.")
        ->take_all();
    serveCmd->add_option("--host", serve.host, "Address the public listener binds")
        ->capture_default_str();
    serveCmd->add_option("--port", serve.port, "Port the public listener binds")
        ->check(CLI::Range(1, 65535))
        ->capture_default_str();
    serveCmd->add_option("--admin-host", serve.adminHost, "Address the admin listener binds")
        ->capture_default_str();
    serveCmd->add_option("--admin-port", serve.adminPort, "Port the admin listener binds")
        ->check(CLI::Range(1, 65535))
        ->capture_default_str();
    serveCmd->add_option("--threads", serve.threads,
            "Concurrent connections the public listener serves")
        ->check(CLI::Range(std::size_t{1}, std::size_t{1024}))
        ->capture_default_str();
    serveCmd->add_option("--max-batch-rows", serve.maxBatchRows, "Rows one request may carry")
        ->check(CLI::Range(std::size_t{1}, std::size_t{4096}))
        ->capture_default_str();
    serveCmd->add_option("--max-body-bytes", serve.maxBodyBytes, "Largest request body accepted")
        ->capture_default_str();
    serveCmd->add_flag("--strict-ready", serve.strictReady,
            "Refuse to start, and report not ready, while any model failed to load");

    serveCmd->callback([&] {
        if (!parseDefaults(serveDefaults, serve.defaults)) {
            exitCode = ServingCli::kBadOption;
            return;
        }
        exitCode = ServingCli::Serve(std::cout, std::cerr, serve);
    });

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    return exitCode;
}
