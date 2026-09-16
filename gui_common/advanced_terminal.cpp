#include "gui_common/advanced_terminal.h"

#include <QScrollBar>


AdvancedTerminal::AdvancedTerminal(QWidget* parent)
    : QPlainTextEdit(parent)
{
    setReadOnly(true);
    setWordWrapMode(QTextOption::NoWrap);
    setFont(QFont("Consolas", 10));

    QPalette p = palette();
    p.setColor(QPalette::Base, QColor(30, 30, 30));
    p.setColor(QPalette::Text, QColor(220, 220, 220));
    setPalette(p);
}

void AdvancedTerminal::write(const QString& text)
{
    QString processedText = text;

    if (processedText.contains('\r')) {
        processedText = processCarriageReturns(processedText);
    }
    if (processedText.contains('\b')) {
        processedText = processBackspaces(processedText);
    }
    processedText = processedText.replace("\t", "    ");

    QScrollBar* scrollBar = verticalScrollBar();
    bool atBottom = scrollBar->value() >= scrollBar->maximum();
    int savedValue = scrollBar->value();

    QTextCursor cursor(document());
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(processedText);

    if (atBottom) {
        scrollBar->setValue(scrollBar->maximum());
    } else {
        // Restore immediately and also after deferred layout updates fire
        scrollBar->setValue(savedValue);
        QMetaObject::invokeMethod(this, [this, savedValue]() {
            verticalScrollBar()->setValue(savedValue);
        }, Qt::QueuedConnection);
    }
}

void AdvancedTerminal::writeLine(const QString& text) {
    write(text + "\n");
}

void AdvancedTerminal::clearLine() {
    QTextCursor cursor = textCursor();
    cursor.movePosition(QTextCursor::StartOfLine);
    cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
    cursor.removeSelectedText();
    setTextCursor(cursor);
}

QString AdvancedTerminal::processCarriageReturns(const QString& text)
{
    QString result;
    QStringList parts = text.split('\r');

    for (int i = 0; i < parts.size(); ++i) {
        if (i == parts.size() - 1) {
            result += parts[i];
        } else {
            // For carriage return, we replace the current line
            QTextCursor cursor(document());
            cursor.movePosition(QTextCursor::End);
            cursor.movePosition(QTextCursor::StartOfLine);
            cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
            cursor.removeSelectedText();
            cursor.insertText(parts[i]);
        }
    }

    return result;
}

QString AdvancedTerminal::processBackspaces(const QString& text)
{
    QString result;
    for (int i = 0; i < text.length(); ++i) {
        if (text[i] == '\b' && !result.isEmpty()) {
            result.chop(1);
        } else {
            result += text[i];
        }
    }
    return result;
}

std::unique_ptr<ThreadSafeTerminalOStream> createTerminalOStream(AdvancedTerminal* terminal) {
    return std::make_unique<ThreadSafeTerminalOStream>(
        std::make_unique<AdvancedTerminalStream>(terminal));
}
