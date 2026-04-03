#pragma once

#include <QValidator>
#include <QRegularExpression>

class CommaSeparatedIntsValidator : public QValidator
{
public:
    CommaSeparatedIntsValidator(QObject *parent = nullptr) : QValidator(parent) {}

    State validate(QString &input, int &pos) const override
    {
        if (input.contains(QRegularExpression(R"([^0-9\s,\-])")))
            return Invalid;

        const QStringList parts = input.split(',');
        QStringList stripped;
        for (const QString &p : parts)
            stripped << p.trimmed();

        for (int i = 0; i < stripped.size(); ++i) {
            if (stripped[i].isEmpty() && i != stripped.size() - 1)
                return Invalid;
        }

        bool endsWithComma = input.trimmed().endsWith(',');

        auto isCompleteInt = [](const QString &s) -> bool {
            return QRegularExpression(R"(^-?\d+$)").match(s).hasMatch();
        };
        auto isPartialInt = [](const QString &s) -> bool {
            return QRegularExpression(R"(^-?\d*$)").match(s).hasMatch();
        };

        if (!endsWithComma && !stripped.isEmpty()) {
            bool allComplete = true;
            for (const QString &p : stripped) {
                if (!isCompleteInt(p)) {
                    allComplete = false;
                    break;
                }
            }
            if (allComplete)
                return Acceptable;
        }

        for (int i = 0; i < stripped.size(); ++i) {
            if (i == stripped.size() - 1) { // last part
                if (!stripped[i].isEmpty() && !isPartialInt(stripped[i]))
                    return Invalid;
            } else {
                if (stripped[i].isEmpty() || !isCompleteInt(stripped[i]))
                    return Invalid;
            }
        }

        return Intermediate;
    }
};
