#pragma once

#include <cstddef>
#include <map>
#include <string>

namespace ServingCli {

struct ListOptions {
    std::string root;
    std::map<std::string, std::string> defaults;
};

struct ServeOptions {
    std::string root;
    std::map<std::string, std::string> defaults;

    std::string host = "127.0.0.1";
    int port = 8080;
    std::string adminHost = "127.0.0.1";
    int adminPort = 8081;

    std::size_t threads = 32;
    std::size_t maxBatchRows = 32;
    std::size_t maxBodyBytes = 8u << 20;

    bool strictReady = false;
};

}  // namespace ServingCli
