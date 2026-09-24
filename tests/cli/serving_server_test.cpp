#include <doctest/doctest.h>

#include "cli/serving/server.h"

#include "core/serving/registry.h"

#include "tests/support/model_dir.h"
#include "tests/support/temp_dir.h"

#include <nlohmann/json.hpp>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// A client small enough to keep httplib in the single translation unit that
// already pays fourteen seconds for it. Every request asks for Connection: close,
// so reading to EOF is the whole response and there is no framing to get wrong.
struct Response {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

std::string Lowered(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

Response Fetch(
    const int port,
    const std::string& method,
    const std::string& path,
    const std::string& contentType = {},
    const std::string& body = {}
) {
    const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(sock >= 0);

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::connect(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);

    std::ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n"
            << "Host: 127.0.0.1\r\n"
            << "Connection: close\r\n";
    if (!contentType.empty()) {
        request << "Content-Type: " << contentType << "\r\n";
    }
    request << "Content-Length: " << body.size() << "\r\n\r\n" << body;

    const std::string text = request.str();
    std::size_t sent = 0;
    while (sent < text.size()) {
        const auto wrote = ::send(sock, text.data() + sent, text.size() - sent, 0);
        if (wrote <= 0) {
            break;
        }
        sent += static_cast<std::size_t>(wrote);
    }

    std::string raw;
    char buffer[4096];
    for (;;) {
        const auto got = ::recv(sock, buffer, sizeof(buffer), 0);
        if (got <= 0) {
            break;
        }
        raw.append(buffer, static_cast<std::size_t>(got));
    }
    ::close(sock);

    Response response;
    const auto split = raw.find("\r\n\r\n");
    REQUIRE(split != std::string::npos);
    response.body = raw.substr(split + 4);

    std::istringstream head(raw.substr(0, split));
    std::string line;
    std::getline(head, line);
    response.status = std::stoi(line.substr(line.find(' ') + 1, 4));
    while (std::getline(head, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        auto value = line.substr(colon + 1);
        while (!value.empty() && value.front() == ' ') {
            value.erase(value.begin());
        }
        response.headers[Lowered(line.substr(0, colon))] = value;
    }
    return response;
}

ServingCli::ServeOptions Options(const std::filesystem::path& root) {
    ServingCli::ServeOptions options;
    options.root = root.string();
    options.defaults = {{"mnist", "v3"}};
    options.port = 0;          // ephemeral, so nothing in the suite reserves a number
    options.adminPort = 0;
    options.threads = 2;
    return options;
}

std::string JsonRow(const std::size_t width) {
    std::vector<double> row(width);
    for (std::size_t i = 0; i < width; ++i) {
        row[i] = static_cast<double>((i * 37) % 251) / 251.;
    }
    return nlohmann::json{{"inputs", std::vector<std::vector<double>>{row}}}.dump();
}

}  // namespace

TEST_CASE("The listeners answer on ephemeral ports and stop when told") {
    Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {
        {"mnist-v1", "mnist", "v1", {64, 16, 3}},
        {"mnist-v3", "mnist", "v3", {64, 16, 3}},
    });

    Serving::RegistryConfig config;
    config.root = root;
    config.defaults = {{"mnist", "v3"}};
    Serving::ModelRegistry registry(config);
    registry.rebuild();

    std::ostringstream err;
    ServingCli::ServerHandle server(Options(root), registry, err);
    REQUIRE(server.start());

    CHECK(server.publicPort() > 0);
    CHECK(server.adminPort() > 0);
    CHECK(server.publicPort() != server.adminPort());

    const int port = server.publicPort();

    SUBCASE("healthz on both listeners") {
        CHECK(Fetch(port, "GET", "/healthz").status == 200);
        CHECK(Fetch(server.adminPort(), "GET", "/healthz").status == 200);
    }

    SUBCASE("the model list routes to the handler") {
        const auto response = Fetch(port, "GET", "/v1/models");
        CHECK(response.status == 200);
        const auto body = nlohmann::json::parse(response.body);
        CHECK(body.at("models").size() == 2);
        CHECK(body.at("models")[1].at("isDefault") == true);
    }

    SUBCASE("the version index and one version route to their handlers") {
        CHECK(Fetch(port, "GET", "/v1/models/mnist").status == 200);
        const auto one = Fetch(port, "GET", "/v1/models/mnist/versions/v3");
        CHECK(one.status == 200);
        CHECK(nlohmann::json::parse(one.body).at("version") == "v3");
    }

    SUBCASE("predict on the default route reports which version answered") {
        const auto response = Fetch(port, "POST", "/v1/models/mnist/predict",
                                    "application/json", JsonRow(64));
        CHECK(response.status == 200);
        CHECK(response.headers.at("x-model-version") == "v3");
        CHECK(nlohmann::json::parse(response.body).at("outputs").size() == 1);
    }

    SUBCASE("predict on a named version reports that version") {
        const auto response = Fetch(port, "POST", "/v1/models/mnist/versions/v1/predict",
                                    "application/json", JsonRow(64));
        CHECK(response.status == 200);
        CHECK(response.headers.at("x-model-version") == "v1");
    }

    SUBCASE("a wrong width is refused over the wire too") {
        const auto response = Fetch(port, "POST", "/v1/models/mnist/predict",
                                    "application/json", JsonRow(65));
        CHECK(response.status == 400);
        CHECK(nlohmann::json::parse(response.body).at("error").at("code") == "bad_input_size");
    }

    SUBCASE("a route miss carries the same envelope as a handler refusal") {
        const auto response = Fetch(port, "GET", "/nope");
        CHECK(response.status == 404);
        CHECK(response.headers.at("content-type").rfind("application/json", 0) == 0);
        CHECK(nlohmann::json::parse(response.body).at("error").at("code") == "not_found");
    }

    SUBCASE("a name the manifest grammar forbids never reaches a snapshot") {
        CHECK(Fetch(port, "GET", "/v1/models/MNIST").status == 404);
        CHECK(Fetch(port, "GET", "/v1/models/-leading-dash").status == 404);
    }

    SUBCASE("readiness and reload live on the admin listener only") {
        const auto ready = Fetch(server.adminPort(), "GET", "/readyz");
        CHECK(ready.status == 200);
        const auto body = nlohmann::json::parse(ready.body);
        CHECK(body.at("ready") == true);
        CHECK(body.at("models") == 2);
        CHECK(body.at("generation") == 1);

        // Absolute server paths and manifest text go out here, and an
        // unauthenticated rebuild is triggerable here, so neither is on the
        // public socket. With no auth in this step the bind address is the only
        // access control there is.
        CHECK(Fetch(port, "GET", "/readyz").status == 404);
        CHECK(Fetch(port, "POST", "/admin/reload").status == 404);
    }

    SUBCASE("a reload over the admin socket publishes the next generation") {
        const auto reload = Fetch(server.adminPort(), "POST", "/admin/reload");
        CHECK(reload.status == 200);
        CHECK(nlohmann::json::parse(reload.body).at("generation") == 2);

        const auto after = Fetch(server.adminPort(), "GET", "/readyz");
        CHECK(nlohmann::json::parse(after.body).at("generation") == 2);
    }

    server.stop();
    server.stop();   // idempotent, so a handle can be stopped and then destroyed
    CHECK(err.str().empty());
}

TEST_CASE("A body over the cap is refused before a handler runs") {
    Tests::TempDir dir;
    const auto root = Tests::WriteModelTree(dir, {{"mnist-v3", "mnist", "v3", {64, 16, 3}}});

    Serving::RegistryConfig config;
    config.root = root;
    Serving::ModelRegistry registry(config);
    registry.rebuild();

    auto options = Options(root);
    options.maxBodyBytes = 1024;

    std::ostringstream err;
    ServingCli::ServerHandle server(options, registry, err);
    REQUIRE(server.start());

    const auto response = Fetch(server.publicPort(), "POST", "/v1/models/mnist/versions/v3/predict",
                                "application/octet-stream", std::string(8192, '\0'));
    CHECK(response.status == 413);
    CHECK(nlohmann::json::parse(response.body).at("error").at("code") == "payload_too_large");

    server.stop();
}

TEST_CASE("The listeners are configured with TCP_NODELAY") {
    // Measured, all four combinations of the flag on a keep-alive binary round
    // trip: server on / client on 91.9us, server on / client off 84.1us, server
    // off / client on 41162us, server off / client off 41324us. The forty
    // milliseconds are the server's alone, and a client cannot buy them back.
    // Nothing about a passing suite or a readable diff would show that, so the
    // setting is pinned here rather than left to a latency measurement that
    // cannot live in CI.
    ServingCli::ServeOptions options;
    options.threads = 7;
    options.maxBodyBytes = 4096;

    const auto publicSide = ServingCli::PublicSettings(options);
    CHECK(publicSide.tcpNoDelay);
    CHECK(publicSide.reuseAddressRatherThanPort);
    CHECK(publicSide.threads == 7);
    CHECK(publicSide.maxThreads == 7);
    CHECK(publicSide.maxQueuedRequests == 0);
    CHECK(publicSide.readTimeoutSeconds == 5);
    CHECK(publicSide.writeTimeoutSeconds == 5);
    CHECK(publicSide.keepAliveTimeoutSeconds == 5);
    CHECK(publicSide.keepAliveMaxCount == 100);
    CHECK(publicSide.maxBodyBytes == 4096);

    // The serve default the handler layer inherits, pinned next to the rest.
    CHECK(ServingCli::ServeOptions{}.maxBatchRows == 32);
    CHECK(ServingCli::ServeOptions{}.maxBodyBytes == 8u << 20);
    // Loopback, not 0.0.0.0: with no auth and no TLS in this step the bind address
    // is the only access control, and a service that listens to the world by
    // default is one that gets deployed "just to try" and stays.
    CHECK(ServingCli::ServeOptions{}.host == "127.0.0.1");
    CHECK(ServingCli::ServeOptions{}.adminHost == "127.0.0.1");

    const auto adminSide = ServingCli::AdminSettings(options);
    CHECK(adminSide.tcpNoDelay);
    // Two, and this is load-bearing rather than a round number. A pool task is a
    // whole connection, so at one thread two admin reloads cannot overlap: the
    // second waits in the backlog, finds the gate free, and re-reads the tree the
    // first one just read. Measured on twelve 5MB models, rebuild 126ms: at one
    // thread both requests answered 200 and the pair took 280ms; at two, the
    // second answered 409 and the pair took 136ms. Drop this to one and the 409
    // documented in the README stops existing over HTTP.
    CHECK(adminSide.threads == 2);
    CHECK(adminSide.maxThreads == 2);
    // The public side keeps its stall detector; a reload needs room to answer.
    CHECK(adminSide.writeTimeoutSeconds == 60);
    CHECK(publicSide.writeTimeoutSeconds == 5);
}
