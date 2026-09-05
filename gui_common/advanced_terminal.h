#pragma once

#include <QPlainTextEdit>
#include <ios>
#include <qtmetamacros.h>
#include <sstream>
#include <mutex>


class AdvancedTerminal : public QPlainTextEdit {
    Q_OBJECT
public:
    AdvancedTerminal(QWidget* parent = nullptr);

    Q_INVOKABLE void write(const QString& text);
    Q_INVOKABLE void writeLine(const QString& text);
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
        std::lock_guard<std::mutex> lock(bufferMutex);
        buffer << t;
        // Flush every completed line: std::endl reaches a streambuf as '\n'
        // plus sync(), never as a manipulator, so output that ends lines with a
        // plain '\n' (the report printers do) would otherwise stay invisible.
        checkAndFlush();
        return *this;
    }

    AdvancedTerminalStream& operator<<(std::ostream& (*manipulator)(std::ostream&)) {
        std::lock_guard<std::mutex> lock(bufferMutex);
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
        std::lock_guard<std::mutex> lock(bufferMutex);
        flushBuffer(false);
    }
private:
    void flushBuffer(const bool addNewline) {
        if (!terminal) {
            return;
        }

        const auto& text = QString::fromStdString(buffer.str());
        if (!text.isEmpty() || addNewline) {
            const auto method = addNewline ? &AdvancedTerminal::writeLine : &AdvancedTerminal::write;
            auto term = this->terminal;
            QMetaObject::invokeMethod(term, [term, method, text]() {
                (term->*method)(text);
            });
            buffer.str("");
            buffer.clear();
        }
    }

    void checkAndFlush() {
        const auto& content = buffer.str();
        // Up to the LAST newline: a chunk carrying several line breaks must
        // not leave its tail invisible until the next write arrives.
        const auto newLinePosition = content.rfind('\n');

        if (newLinePosition != std::string::npos) {
            const auto& toFlush = QString::fromStdString(content.substr(0, newLinePosition + 1));
            auto term = this->terminal;
            QMetaObject::invokeMethod(term, [term, toFlush]() {
                term->write(toFlush);
            });
            const auto& remaining = content.substr(newLinePosition + 1);
            buffer.str(remaining);
            buffer.clear();
        }
    }

    AdvancedTerminal* terminal;
    std::mutex bufferMutex;
    std::ostringstream buffer;
};


class ThreadSafeTerminalBuffer : public std::streambuf
{
public:
    ThreadSafeTerminalBuffer(AdvancedTerminalStream* stream) : stream(stream) {}

protected:
    std::streamsize xsputn(const char_type* s, std::streamsize count) override {
        if (stream) {
            std::string string(s, count);
            *stream << string;
        }
        return count;
    }

    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof() && stream) {
            *stream << static_cast<char>(ch);
        }
        return ch;
    }

    int sync() override {
        if (stream) {
            stream->flush();
        }
        return 0;
    }

private:
    AdvancedTerminalStream* stream;
};


class ThreadSafeTerminalOStream : public std::ostream
{
public:
    ThreadSafeTerminalOStream(AdvancedTerminalStream* stream)
        : std::ostream(&buffer)
        , buffer(stream)
    {}

    ThreadSafeTerminalOStream(const ThreadSafeTerminalOStream&) = delete;
    ThreadSafeTerminalOStream& operator=(const ThreadSafeTerminalOStream&) = delete;

    ThreadSafeTerminalOStream(ThreadSafeTerminalOStream&& other) noexcept
        : std::ostream(std::move(other))
        , buffer(std::move(other.buffer)) {
        rdbuf(&buffer);
    }

    ThreadSafeTerminalOStream& operator=(ThreadSafeTerminalOStream&& other) noexcept {
        if (this != &other) {
            std::ostream::operator=(std::move(other));
            buffer = std::move(other.buffer);
            rdbuf(&buffer);
        }
        return *this;
    }

private:
    ThreadSafeTerminalBuffer buffer;
};


std::unique_ptr<ThreadSafeTerminalOStream> createTerminalOStream(AdvancedTerminal* terminal);
