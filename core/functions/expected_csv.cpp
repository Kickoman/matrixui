#include "core/functions/expected_csv.h"

#include <algorithm>
#include <format>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace Genetizer {

namespace {

std::string Trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = text.find_last_not_of(" \t");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitLine(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    std::vector<std::string> cells;
    std::size_t begin = 0;
    while (true) {
        const auto comma = line.find(',', begin);
        cells.push_back(Trim(line.substr(begin, comma - begin)));
        if (comma == std::string::npos) {
            return cells;
        }
        begin = comma + 1;
    }
}

[[noreturn]] static void ThrowNotANumber(const std::string& cell, std::size_t lineNumber,
                                         const std::string& column) {
    throw std::runtime_error(
        "line " + std::to_string(lineNumber) + ", column '" + column +
        "': not a number: '" + cell + "'");
}

double ParseNumber(const std::string& cell, std::size_t lineNumber, const std::string& column) {
    std::string_view s = cell;

    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        ThrowNotANumber(cell, lineNumber, column);
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    s = s.substr(first, last - first + 1);

    if (s.front() == '+') {
        s.remove_prefix(1);
        if (s.empty() || s.front() == '-') {
            ThrowNotANumber(cell, lineNumber, column);
        }
    }

    double value{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        ThrowNotANumber(cell, lineNumber, column);
    }
    return value;
}

}  // namespace

std::vector<Entry> ParseExpectedCsv(std::istream& in) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty CSV: expected a header row");
    }

    const auto header = SplitLine(line);
    if (header.size() < 2) {
        throw std::runtime_error(
            "CSV needs at least one variable column and 'expected', got header: " + line);
    }
    if (header.back() != "expected") {
        throw std::runtime_error(
            "CSV header must end with an 'expected' column, got: " + line);
    }
    for (std::size_t i = 0; i + 1 < header.size(); ++i) {
        if (header[i].empty()) {
            throw std::runtime_error("CSV header has an empty variable name: " + line);
        }
    }

    std::vector<Entry> entries;
    std::size_t lineNumber = 1;
    while (std::getline(in, line)) {
        ++lineNumber;
        auto cells = SplitLine(line);
        if (cells.size() == 1 && cells[0].empty()) {
            continue;  // blank line
        }
        if (cells.size() != header.size()) {
            throw std::runtime_error(
                "line " + std::to_string(lineNumber) + ": expected " +
                std::to_string(header.size()) + " columns, got " + std::to_string(cells.size()));
        }

        Entry entry;
        for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
            entry.variables.push_back(Variable{
                .name = header[i],
                .value = ParseNumber(cells[i], lineNumber, header[i]),
            });
        }
        entry.expectedResult = ParseNumber(cells.back(), lineNumber, header.back());
        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        throw std::runtime_error("CSV has no data rows");
    }
    return entries;
}

void WriteExpectedCsv(std::ostream& out,
                      const std::vector<std::string>& variableNames,
                      const std::vector<Entry>& entries) {
    for (const auto& name : variableNames) {
        out << name << ',';
    }
    out << "expected\n";

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        for (const auto& name : variableNames) {
            const auto found = std::find_if(
                entry.variables.cbegin(), entry.variables.cend(),
                [&name](const Variable& variable) { return variable.name == name; });
            if (found == entry.variables.cend()) {
                throw std::runtime_error(
                    "point " + std::to_string(i + 1) + " has no value for '" + name + "'");
            }
            out << std::format("{}", found->value) << ',';
        }
        out << std::format("{}", entry.expectedResult) << '\n';
    }
}

}  // namespace Genetizer
