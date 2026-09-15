#include "cli/functions/csv.h"

#include <istream>
#include <stdexcept>
#include <string>

namespace FunctionsCli {

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

double ParseNumber(const std::string& cell, std::size_t lineNumber, const std::string& column) {
    const auto fail = [&] {
        throw std::runtime_error(
            "line " + std::to_string(lineNumber) + ", column '" + column +
            "': not a number: '" + cell + "'");
    };
    try {
        std::size_t consumed = 0;
        const auto value = std::stod(cell, &consumed);
        if (consumed != cell.size()) {
            fail();
        }
        return value;
    } catch (const std::invalid_argument&) {
        fail();
    } catch (const std::out_of_range&) {
        fail();
    }
    return 0;  // unreachable
}

}  // namespace

std::vector<Genetizer::Entry> ParseExpectedCsv(std::istream& in) {
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

    std::vector<Genetizer::Entry> entries;
    std::size_t lineNumber = 1;
    while (std::getline(in, line)) {
        ++lineNumber;
        auto cells = SplitLine(line);
        if (cells.size() == 1 && cells[0].empty()) {
            continue;
        }
        if (cells.size() != header.size()) {
            throw std::runtime_error(
                "line " + std::to_string(lineNumber) + ": expected " +
                std::to_string(header.size()) + " columns, got " + std::to_string(cells.size()));
        }

        Genetizer::Entry entry;
        for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
            entry.variables.push_back(Genetizer::Variable{
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

}  // namespace FunctionsCli
