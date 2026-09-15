#pragma once

#include <algorithm>
#include <ranges>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <format>


namespace tabs {

template <class...> inline constexpr bool always_false = false;


class Row {
public:
    std::size_t size() const { return values.size(); }
    const std::string& operator[](std::size_t index) const { return values[index]; }

    template<class T>
    Row& operator<<(const T& value) {
        if constexpr (std::is_same_v<T, char>) {
            values.emplace_back(1, value);
        } else if constexpr (std::is_convertible_v<T, std::string>) {
            values.emplace_back(value);
        } else if constexpr (std::is_floating_point_v<T>) {
            values.push_back(std::format("{:.4f}", value));
        } else if constexpr (requires { std::to_string(value); }) {
            values.push_back(std::to_string(value));
        } else {
            static_assert(always_false<T>, "The type should be convertible to std::string or have and std::to_string specification");
        }
        return *this;
    }

    auto begin() { return values.begin(); }
    auto begin() const { return values.begin(); }
    auto end() { return values.end(); }
    auto end() const { return values.end(); }
private:
    std::vector<std::string> values;
};

class Tabulator {
public:
    Row& addRow() {
        rows.push_back({});
        return rows.back();
    }

    Row& addHeader() {
        header = {};
        return header;
    }

    std::string tabulate() const {
        const std::size_t maxColumnCount = std::ranges::max(rows | std::views::transform(&Row::size));

        std::vector<size_t> columnWidth(maxColumnCount);
        std::ranges::transform(header, columnWidth.begin(), &std::string::size);

        for (const auto& row : rows) {
            for (std::size_t i = 0; i < row.size(); ++i) {
                columnWidth[i] = std::max(columnWidth[i], row[i].size());
            }
        }

        std::stringstream stream;
        for (std::size_t i = 0; i < maxColumnCount; ++i) {
            stream << std::setw(columnWidth[i])
                   << (i < header.size() ? header[i] : " ");
            if (i + 1 < maxColumnCount) {
                stream << std::setw(0) << " | ";
            }
        }
        stream << "\n";

        for (std::size_t i = 0; i < maxColumnCount; ++i) {
            stream << std::string(columnWidth[i] + 3 * static_cast<int>(i + 1 < maxColumnCount), '=');
        }
        stream << "\n";

        for (std::size_t rowIdx = 0; rowIdx < rows.size(); ++rowIdx) {
            const auto& row = rows[rowIdx];
            for (std::size_t i = 0; i < maxColumnCount; ++i) {
                stream << std::setw(columnWidth[i])
                       << (i < row.size() ? row[i] : " ");
                if (i + 1 < maxColumnCount) {
                    stream << std::setw(0) << " | ";
                }
            }
            if (rowIdx + 1 < rows.size()) {
                stream << "\n";
            }
        }
        return stream.str();
    }

private:
    Row header;
    std::vector<Row> rows;
};

}
