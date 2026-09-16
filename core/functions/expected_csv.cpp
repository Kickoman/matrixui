#include "core/functions/expected_csv.h"

#include "core/lib/text.h"

#include <algorithm>
#include <format>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace Genetizer {

namespace {

std::vector<std::string> SplitCells(const std::string& line) {
    auto cells = Text::Split(line, ",");
    for (auto& cell : cells) {
        cell = std::string(Text::Trim(cell));
    }
    return cells;
}

[[noreturn]] void ThrowNotANumber(const std::string& cell, std::size_t lineNumber,
                                  const std::string& column) {
    throw std::runtime_error(
        "line " + std::to_string(lineNumber) + ", column '" + column +
        "': not a number: '" + cell + "'");
}

double ParseNumber(const std::string& cell, std::size_t lineNumber, const std::string& column) {
    const auto value = Text::ParseNumber<double>(cell);
    if (!value.has_value()) {
        ThrowNotANumber(cell, lineNumber, column);
    }
    return *value;
}

}  // namespace

std::vector<Entry> ParseExpectedCsv(std::istream& in) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty CSV: expected a header row");
    }

    const auto header = SplitCells(line);
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
        auto cells = SplitCells(line);
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
