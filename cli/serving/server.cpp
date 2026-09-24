#include "cli/serving/server.h"

#include "cli/serving/commands.h"
#include "cli/serving/handlers.h"

#include "core/serving/registry.h"
#include "core/serving/snapshot.h"

#include <httplib/httplib.h>

#include <fcntl.h>
#include <unistd.h>

#include <csignal>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>

namespace ServingCli {

namespace {

// Verbatim the identifier grammar ParseManifest enforces, so a path that cannot
// name a model is refused by the router without touching a snapshot. It is not a
// second truth: a name failing this can never be in a composition.
constexpr char kName[] = R"(/v1/models/([a-z0-9][a-z0-9_-]{0,63}))";
constexpr char kVersion[] = R"(/versions/([a-z0-9][a-z0-9_-]{0,63}))";

std::string Captured(const httplib::Request& request, const std::size_t at) {
    return request.matches[at].str();
}

void Apply(const HttpReply& reply, httplib::Response& response) {
    response.status = reply.status;
    if (!reply.modelVersion.empty()) {
        response.set_header("X-Model-Version", reply.modelVersion);
    }
    response.set_content(reply.body, reply.contentType);
}

std::string_view CodeForStatus(const int status) {
    switch (status) {
        case 404: return "not_found";
        case 405: return "method_not_allowed";
        case 413: return "payload_too_large";
        case 414: return "uri_too_long";
        case 431: return "request_header_fields_too_large";
        default: return status >= 500 ? "internal" : "bad_request";
    }
}

void ApplySettings(httplib::Server& server, const ServerSettings& settings) {
    server.set_tcp_nodelay(settings.tcpNoDelay);

    if (settings.reuseAddressRatherThanPort) {
        // httplib's default_socket_options sets SO_REUSEPORT on Linux, which lets
        // a second process bind a port this one already serves: the kernel then
        // splits requests between two registries. SO_REUSEADDR still allows a
        // restart over TIME_WAIT but refuses a live port.
        server.set_socket_options([](socket_t sock) {
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        });
    }

    // A pool task is a whole connection, not a request, so this is a ceiling on
    // concurrent clients rather than on concurrent work.
    const auto threads = settings.threads;
    const auto maxThreads = settings.maxThreads;
    const auto queued = settings.maxQueuedRequests;
    const auto idle = settings.idleTimeoutSeconds;
    server.new_task_queue = [threads, maxThreads, queued, idle] {
        return new httplib::ThreadPool(threads, maxThreads, queued, idle);
    };

    server.set_read_timeout(settings.readTimeoutSeconds, 0);
    server.set_write_timeout(settings.writeTimeoutSeconds, 0);
    server.set_keep_alive_timeout(settings.keepAliveTimeoutSeconds);
    server.set_keep_alive_max_count(settings.keepAliveMaxCount);
    server.set_payload_max_length(settings.maxBodyBytes);
}

void InstallHooks(httplib::Server& server, std::ostream& err) {
    // Without this a handler that throws gets a bodyless 500 from the library;
    // mapping is by type, never by message text.
    server.set_exception_handler([&err](const httplib::Request& request,
                                       httplib::Response& response,
                                       std::exception_ptr raised) {
        std::string what = "unknown";
        try {
            std::rethrow_exception(raised);
        } catch (const std::exception& error) {
            what = error.what();
        } catch (...) {
        }
        // The message can carry absolute paths, so it goes to the operator's
        // stream and never to the client.
        err << "handler for " << request.path << " threw: " << what << "\n";
        const auto reply = ErrorReply(500, "internal", "internal error");
        response.status = reply.status;
        response.set_content(reply.body, reply.contentType);
    });

    // Backfills the statuses httplib raises itself -- a route miss, a body over
    // the cap -- without clobbering a body one of our handlers already wrote.
    server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (!response.body.empty()) {
            return;
        }
        const auto reply = ErrorReply(response.status, CodeForStatus(response.status),
                                     "request refused");
        response.set_content(reply.body, reply.contentType);
    });

    // The only hook that fires when the pool refuses a connection: there httplib
    // closes the socket without writing, so nothing else can see it. The Request
    // is null on that path, so a refusal is countable but not attributable.
    server.set_error_logger([&err](const httplib::Error& error, const httplib::Request* request) {
        err << "connection error: " << httplib::to_string(error);
        if (request != nullptr) {
            err << " on " << request->path;
        }
        err << "\n";
    });
}

void InstallPublicRoutes(
    httplib::Server& server,
    Serving::ModelRegistry& registry,
    const HandlerLimits& limits
) {
    server.Get("/healthz", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ok\"}\n", "application/json");
    });

    server.Get("/v1/models", [&registry](const httplib::Request&, httplib::Response& response) {
        Apply(ListModels(*registry.snapshot()), response);
    });

    server.Get(kName, [&registry](const httplib::Request& request, httplib::Response& response) {
        Apply(DescribeModel(*registry.snapshot(), Captured(request, 1)), response);
    });

    server.Get(std::string(kName) + kVersion,
               [&registry](const httplib::Request& request, httplib::Response& response) {
        Apply(DescribeVersion(*registry.snapshot(), Captured(request, 1), Captured(request, 2)),
              response);
    });

    server.Post(std::string(kName) + "/predict",
                [&registry, limits](const httplib::Request& request, httplib::Response& response) {
        const auto snapshot = registry.snapshot();
        Apply(Predict(*snapshot, PredictRequest{
            Captured(request, 1), {}, request.get_header_value("Content-Type"), request.body,
        }, limits), response);
    });

    server.Post(std::string(kName) + kVersion + "/predict",
                [&registry, limits](const httplib::Request& request, httplib::Response& response) {
        const auto snapshot = registry.snapshot();
        Apply(Predict(*snapshot, PredictRequest{
            Captured(request, 1), Captured(request, 2),
            request.get_header_value("Content-Type"), request.body,
        }, limits), response);
    });
}

void InstallAdminRoutes(
    httplib::Server& server,
    Serving::ModelRegistry& registry,
    ReloadGate& gate,
    const bool strictReady
) {
    server.Get("/healthz", [](const httplib::Request&, httplib::Response& response) {
        // Liveness never reads the registry: a probe that can fail for a reason a
        // restart will not fix is a probe that gets the process killed.
        response.set_content("{\"status\":\"ok\"}\n", "application/json");
    });

    server.Get("/readyz", [&registry, strictReady](const httplib::Request&, httplib::Response& response) {
        Apply(Readyz(*registry.snapshot(), strictReady), response);
    });

    server.Post("/admin/reload", [&registry, &gate](const httplib::Request&, httplib::Response& response) {
        Apply(Reload(registry, gate), response);
    });
}

// The write end of the self-pipe. A signal handler may touch nothing else: no
// mutex, no allocation, no atomic<shared_ptr>, no stream. write() is on the
// async-signal-safe list, and one byte through a pipe is the whole mechanism.
volatile sig_atomic_t gSignalWriteEnd = -1;

extern "C" void OnSignal(const int number) {
    const unsigned char byte = number == SIGHUP ? 1 : 0;
    const ssize_t wrote = ::write(static_cast<int>(gSignalWriteEnd), &byte, 1);
    (void)wrote;
}

bool InstallSignalHandlers() {
    struct sigaction action{};
    action.sa_handler = &OnSignal;
    action.sa_flags = SA_RESTART;
    sigemptyset(&action.sa_mask);
    return ::sigaction(SIGHUP, &action, nullptr) == 0
        && ::sigaction(SIGINT, &action, nullptr) == 0
        && ::sigaction(SIGTERM, &action, nullptr) == 0;
}

// Ignored rather than restored to the default: shutdown is already under way, and
// SIG_DFL for SIGHUP would turn a stray signal into termination by signal instead
// of a clean exit status.
void SilenceSignalHandlers() {
    struct sigaction action{};
    action.sa_handler = SIG_IGN;
    sigemptyset(&action.sa_mask);
    ::sigaction(SIGHUP, &action, nullptr);
    ::sigaction(SIGINT, &action, nullptr);
    ::sigaction(SIGTERM, &action, nullptr);
}

HandlerLimits LimitsOf(const ServeOptions& options) {
    return HandlerLimits{options.maxBatchRows};
}

}  // namespace

ServerSettings PublicSettings(const ServeOptions& options) {
    ServerSettings settings;
    settings.threads = options.threads;
    settings.maxThreads = options.threads;
    settings.maxQueuedRequests = 0;
    settings.maxBodyBytes = options.maxBodyBytes;
    return settings;
}

ServerSettings AdminSettings(const ServeOptions& options) {
    ServerSettings settings = PublicSettings(options);
    // One thread: rebuild() serialises on the registry's own mutex anyway, and an
    // admin request must never wait behind a public one.
    // Two, not one, and the reason is the pool's shape: a task is a whole
    // connection, so a single-threaded listener cannot have two admin requests in
    // flight. The second would wait in the backlog until the first rebuild had
    // finished and released the gate, get a thread, see the gate free, and pay for
    // a second full re-read of the tree the first one just read. Two threads make
    // the 409 reachable, which is what turns it from a claim into a guarantee.
    settings.threads = 2;
    settings.maxThreads = 2;
    // A reload re-reads the whole tree, so the admin side gets room to answer
    // where the public side keeps its five-second stall detector.
    settings.writeTimeoutSeconds = 60;
    return settings;
}

struct ServerHandle::State {
    httplib::Server publicListener;
    httplib::Server adminListener;
    std::thread publicThread;
    std::thread adminThread;
    int publicPort = 0;
    int adminPort = 0;
    bool running = false;
    ReloadGate gate;
};

ServerHandle::ServerHandle(
    const ServeOptions& options,
    Serving::ModelRegistry& registry,
    std::ostream& err
)
    : state(std::make_unique<State>()) {
    ApplySettings(state->publicListener, PublicSettings(options));
    ApplySettings(state->adminListener, AdminSettings(options));
    InstallHooks(state->publicListener, err);
    InstallHooks(state->adminListener, err);
    InstallPublicRoutes(state->publicListener, registry, LimitsOf(options));
    InstallAdminRoutes(state->adminListener, registry, state->gate, options.strictReady);
    state->publicPort = options.port;
    state->adminPort = options.adminPort;
}

ServerHandle::~ServerHandle() {
    stop();
}

bool ServerHandle::start() {
    const auto bind = [](httplib::Server& server, const std::string& host, int& port) {
        if (port == 0) {
            port = server.bind_to_any_port(host);
            return port > 0;
        }
        return server.bind_to_port(host, port);
    };

    if (!bind(state->publicListener, "127.0.0.1", state->publicPort)) {
        return false;
    }
    if (!bind(state->adminListener, "127.0.0.1", state->adminPort)) {
        return false;
    }

    state->publicThread = std::thread([this] { state->publicListener.listen_after_bind(); });
    state->adminThread = std::thread([this] { state->adminListener.listen_after_bind(); });
    state->publicListener.wait_until_ready();
    state->adminListener.wait_until_ready();
    state->running = true;
    return true;
}

void ServerHandle::stop() {
    if (!state->running) {
        return;
    }
    state->publicListener.stop();
    state->adminListener.stop();
    state->publicThread.join();
    state->adminThread.join();
    state->running = false;
}

int ServerHandle::publicPort() const {
    return state->publicPort;
}

int ServerHandle::adminPort() const {
    return state->adminPort;
}

int RunServer(
    std::ostream& out,
    std::ostream& err,
    const ServeOptions& options,
    Serving::ModelRegistry& registry
) {
    httplib::Server publicListener;
    httplib::Server adminListener;
    ReloadGate gate;

    ApplySettings(publicListener, PublicSettings(options));
    ApplySettings(adminListener, AdminSettings(options));
    InstallHooks(publicListener, err);
    InstallHooks(adminListener, err);
    InstallPublicRoutes(publicListener, registry, LimitsOf(options));
    InstallAdminRoutes(adminListener, registry, gate, options.strictReady);

    if (!publicListener.bind_to_port(options.host, options.port)) {
        err << "Can't bind " << options.host << ":" << options.port << "\n";
        return kCannotBind;
    }
    if (!adminListener.bind_to_port(options.adminHost, options.adminPort)) {
        err << "Can't bind " << options.adminHost << ":" << options.adminPort << "\n";
        return kCannotBind;
    }

    int signalPipe[2] = {-1, -1};
    if (::pipe2(signalPipe, O_CLOEXEC) != 0) {
        err << "Can't open the signal pipe\n";
        return kCannotBind;
    }
    gSignalWriteEnd = signalPipe[1];
    if (!InstallSignalHandlers()) {
        err << "Can't install the signal handlers\n";
        ::close(signalPipe[0]);
        ::close(signalPipe[1]);
        return kCannotBind;
    }

    out << "Serving on " << options.host << ":" << options.port
        << ", admin on " << options.adminHost << ":" << options.adminPort << "\n";
    out.flush();

    // The rebuild runs here, on a thread of its own, never in the handler: the
    // handler is a signal handler and may do nothing but write a byte.
    std::thread signals([&] {
        for (;;) {
            unsigned char byte = 0;
            const auto got = ::read(signalPipe[0], &byte, 1);
            if (got <= 0) {
                return;
            }
            if (byte == 0) {
                publicListener.stop();
                adminListener.stop();
                return;
            }
            const auto reply = Reload(registry, gate);
            err << "SIGHUP: " << reply.body;
            err.flush();
        }
    });

    std::thread admin([&adminListener] { adminListener.listen_after_bind(); });
    publicListener.listen_after_bind();
    adminListener.stop();

    // Unblocks the reader if the listeners stopped for any other reason. Harmless
    // when it has already returned: the byte sits in a pipe we are about to close.
    const unsigned char stop = 0;
    const ssize_t wrote = ::write(signalPipe[1], &stop, 1);
    (void)wrote;

    signals.join();
    admin.join();

    // Order matters: stop the handlers and drop the descriptor before closing it.
    // The other way round, a signal arriving in the window writes into a closed
    // fd -- or into whatever else has since been handed that number.
    SilenceSignalHandlers();
    gSignalWriteEnd = -1;
    ::close(signalPipe[0]);
    ::close(signalPipe[1]);
    return kSuccess;
}

}  // namespace ServingCli
