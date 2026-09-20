#include "cli/serving/commands.h"

#include "core/serving/build.h"
#include "core/serving/error.h"
#include "core/serving/registry.h"
#include "core/serving/snapshot.h"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace ServingCli {

namespace {

struct Row {
    std::string name;
    std::string version;
    std::string input;
    std::string output;
    std::string integrity;
    std::string isDefault;
};

std::string Describe(const Serving::InputContract& input) {
    std::string text = std::to_string(input.size);
    if (!input.shape.empty()) {
        text += " (" + std::string(Serving::ToString(input.layout));
        for (std::size_t i = 0; i < input.shape.size(); ++i) {
            text += (i == 0 ? " " : "x") + std::to_string(input.shape[i]);
        }
        text += ")";
    }
    return text;
}

std::string Describe(const Serving::OutputContract& output) {
    std::string text = std::to_string(output.size);
    if (output.kind != Serving::OutputKind::Raw) {
        text += " " + std::string(Serving::ToString(output.kind));
    }
    return text;
}

void PrintTable(std::ostream& out, const std::vector<Row>& rows) {
    static const Row header{"MODEL", "VERSION", "INPUT", "OUTPUT", "INTEGRITY", "DEFAULT"};

    std::size_t name = header.name.size();
    std::size_t version = header.version.size();
    std::size_t input = header.input.size();
    std::size_t output = header.output.size();
    std::size_t integrity = header.integrity.size();
    for (const auto& row : rows) {
        name = std::max(name, row.name.size());
        version = std::max(version, row.version.size());
        input = std::max(input, row.input.size());
        output = std::max(output, row.output.size());
        integrity = std::max(integrity, row.integrity.size());
    }

    const auto print = [&](const Row& row) {
        out << std::left
            << std::setw(static_cast<int>(name + 2)) << row.name
            << std::setw(static_cast<int>(version + 2)) << row.version
            << std::setw(static_cast<int>(input + 2)) << row.input
            << std::setw(static_cast<int>(output + 2)) << row.output
            << std::setw(static_cast<int>(integrity + 2)) << row.integrity
            << row.isDefault << "\n";
    };

    print(header);
    for (const auto& row : rows) {
        print(row);
    }
}

int RunList(std::ostream& out, std::ostream& err, const ListOptions& options) {
    Serving::RegistryConfig config;
    config.root = options.root;
    config.defaults = options.defaults;

    const auto snapshot = Serving::Build(config);
    const auto described = Serving::Describe(snapshot);

    std::vector<Row> rows;
    rows.reserve(described.models.size());
    for (const auto& entry : described.models) {
        const auto& manifest = entry.model->manifest();
        rows.push_back(Row{
            manifest.name,
            manifest.version,
            Describe(manifest.input),
            Describe(manifest.output),
            entry.model->integrity() == Serving::IntegrityCheck::Verified ? "verified" : "not declared",
            entry.isDefault ? "yes" : "",
        });
    }

    if (rows.empty()) {
        out << "No models under " << options.root << "\n";
    } else {
        PrintTable(out, rows);
    }

    if (described.failures.empty()) {
        return kSuccess;
    }

    err << "\n" << described.failures.size()
        << (described.failures.size() == 1 ? " problem:\n" : " problems:\n");
    for (const auto& failure : described.failures) {
        std::string reason = failure.reason;
        for (auto at = reason.find('\n'); at != std::string::npos; at = reason.find('\n', at + 5)) {
            reason.replace(at, 1, "\n    ");
        }
        err << "\n  " << failure.directory.string()
            << "\n    " << Serving::ToString(failure.kind) << ": " << reason << "\n";
    }
    return kIncomplete;
}

}  // namespace

int List(std::ostream& out, std::ostream& err, const ListOptions& options) {
    try {
        return RunList(out, err, options);
    } catch (const Serving::Error& error) {
        err << error.what() << "\n";
        return kUnusableRoot;
    } catch (const std::exception& error) {
        err << error.what() << "\n";
        return kUnusableRoot;
    }
}

}  // namespace ServingCli
