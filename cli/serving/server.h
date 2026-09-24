#pragma once

#include "cli/serving/options.h"

#include <cstddef>
#include <iosfwd>
#include <memory>

namespace Serving {
class ModelRegistry;
}

namespace ServingCli {

// Every setting the listeners apply, so a test can assert them without opening a
// socket. TCP_NODELAY above all: httplib defaults it to false, and without it the
// service still answers, ninety times slower. That regression is invisible in a
// diff and invisible in a passing test suite, so it is pinned here.
struct ServerSettings {
    bool tcpNoDelay = true;
    bool reuseAddressRatherThanPort = true;
    std::size_t threads = 0;
    std::size_t maxThreads = 0;
    std::size_t maxQueuedRequests = 0;
    int idleTimeoutSeconds = 3;
    int readTimeoutSeconds = 5;
    int writeTimeoutSeconds = 5;
    int keepAliveTimeoutSeconds = 5;
    std::size_t keepAliveMaxCount = 100;
    std::size_t maxBodyBytes = 0;
};

ServerSettings PublicSettings(const ServeOptions& options);
ServerSettings AdminSettings(const ServeOptions& options);

// Serves until the process is stopped.
int RunServer(
    std::ostream& out,
    std::ostream& err,
    const ServeOptions& options,
    Serving::ModelRegistry& registry
);

// The same listeners, started on their own threads and stoppable, for tests.
// Port 0 in the options binds an ephemeral port, which publicPort() reports, so
// nothing in the suite has to reserve a number. No httplib type escapes here.
class ServerHandle {
public:
    // `err` receives what the hooks report: a handler that threw, a connection
    // the pool refused. A test asserts on it.
    ServerHandle(const ServeOptions& options, Serving::ModelRegistry& registry, std::ostream& err);
    ~ServerHandle();

    ServerHandle(const ServerHandle&) = delete;
    ServerHandle& operator=(const ServerHandle&) = delete;

    bool start();
    void stop();

    int publicPort() const;
    int adminPort() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}  // namespace ServingCli
