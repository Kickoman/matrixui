#pragma once

#include <QPlainTextEdit>
#include <sstream>

class AdvancedTerminal : public QPlainTextEdit {
    Q_OBJECT
public:
    AdvancedTerminal(QWidget* parent = nullptr);

    void write(const QString& text);
    void writeLine(const QString& text);
    void clearLine();

private:
    QString processCarriageReturns(const QString& text);
    QString processBackspaces(const QString& text);
};


class AdvancedTerminalStream {
public:
    AdvancedTerminalStream(AdvancedTerminal* terminal) : terminal(terminal) {}

    template<class T>
    AdvancedTerminalStream& operator<<(const T& t) {
        buffer << t;
        return *this;
    }

    AdvancedTerminalStream& operator<<(std::ostream& (*manipulator)(std::ostream&)) {
        if (manipulator == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
            flushBuffer(true);
        } else if (manipulator == static_cast<std::ostream& (*)(std::ostream&)>(std::flush)) {
            flushBuffer(false);
        } else {
            manipulator(buffer);
            checkAndFlush();
        }
        return *this;
    }

    void flush() {
        flushBuffer(false);
    }
private:
    void flushBuffer(const bool addNewline) {
        if (!terminal) {
            return;
        }

        const auto& text = buffer.str();
        if (!text.empty() || addNewline) {
            if (addNewline) {
                terminal->writeLine(QString::fromStdString(text));
            } else {
                terminal->write(QString::fromStdString(text));
            }
            buffer.str("");
            buffer.clear();
        }
    }

    void checkAndFlush() {
        const auto& content = buffer.str();
        const auto newLinePosition = content.find('\n');

        if (newLinePosition != std::string::npos) {
            const auto& toFlush = content.substr(0, newLinePosition + 1);
            terminal->write(QString::fromStdString(toFlush));

            const auto& remaining = content.substr(newLinePosition + 1);
            buffer.str(remaining);
            buffer.clear();
        }
    }

    AdvancedTerminal* terminal;
    std::ostringstream buffer;
};
