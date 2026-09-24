#include "cli/serving/server.h"

#include "cli/serving/commands.h"

#include <httplib/httplib.h>

#include <ostream>
#include <thread>

namespace ServingCli {

namespace {

void Configure(httplib::Server& server, const std::size_t threads, const std::size_t maxBodyBytes) {
    // CPPHTTPLIB_TCP_NODELAY defaults to false, and Nagle with a delayed ACK
    // costs about forty milliseconds a request. A client has to set it on its
    // own socket too.
    server.set_tcp_nodelay(true);

    // A pool task is a whole connection, not a request, so this is a ceiling on
    // concurrent clients rather than on concurrent work.
    server.new_task_queue = [threads] {
        return new httplib::ThreadPool(threads, threads, 0, 3);
    };

    // httplib's default_socket_options sets SO_REUSEPORT on Linux, which lets a
    // second process bind a port this one is already serving: the kernel then
    // splits requests between two registries. SO_REUSEADDR still allows a
    // restart over TIME_WAIT but refuses a live port, which is what makes
    // kCannotBind mean anything.
    server.set_socket_options([](socket_t sock) {
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
    });

    server.set_read_timeout(5, 0);
    server.set_write_timeout(5, 0);
    server.set_keep_alive_timeout(5);
    server.set_keep_alive_max_count(100);
    server.set_payload_max_length(maxBodyBytes);
}

void Health(httplib::Server& server) {
    server.Get("/healthz", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ok\"}\n", "application/json");
    });
}

bool Bind(std::ostream& err, httplib::Server& server, const std::string& host, const int port) {
    if (server.bind_to_port(host, port)) {
        return true;
    }
    err << "Can't bind " << host << ":" << port << "\n";
    return false;
}

}  // namespace

int RunServer(std::ostream& out, std::ostream& err, const ServeOptions& options) {
    httplib::Server publicListener;
    httplib::Server adminListener;

    Configure(publicListener, options.threads, options.maxBodyBytes);
    Configure(adminListener, 1, options.maxBodyBytes);
    Health(publicListener);
    Health(adminListener);

    if (!Bind(err, publicListener, options.host, options.port)) {
        return kCannotBind;
    }
    if (!Bind(err, adminListener, options.adminHost, options.adminPort)) {
        return kCannotBind;
    }

    out << "Serving on " << options.host << ":" << options.port
        << ", admin on " << options.adminHost << ":" << options.adminPort << "\n";
    out.flush();

    std::thread admin([&adminListener] { adminListener.listen_after_bind(); });
    publicListener.listen_after_bind();

    adminListener.stop();
    admin.join();
    return kSuccess;
}

}  // namespace ServingCli
