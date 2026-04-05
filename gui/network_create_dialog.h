#pragma once

#include <QDialog>
#include "neural_network.h"

class QLabel;
class QLineEdit;
class QPushButton;
class QDialogButtonBox;
class QComboBox;


class NetworkCreateDialog : public QDialog
{
    Q_OBJECT
public:
    explicit NetworkCreateDialog(QWidget* parent = nullptr);

    QString getNetworkName() const;
    Neural::NeuralNetworkConfiguration getConfiguration() const;

    void setCurrentNetworkPath(const QString& path);

private slots:
    void handleNetworkPathButtonClicked();
    void handleSelectedPathChanged(const QString& text);

private:

    QLineEdit* networkPathInput = nullptr;
    QLineEdit* networkLayersInput = nullptr;
    QPushButton* networkPathButton = nullptr;
    QDialogButtonBox* buttonBox = nullptr;

    QComboBox* hiddenActivation = nullptr;
    QComboBox* outputActivation = nullptr;
};
