#pragma once

#include <cstddef>
#include <vector>

#include <QDialog>
#include "core/nn/neural_network.h"

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
    void setDefaultLayersText(const QString& layers);
    void setDefaultLayersText(const std::vector<std::size_t>& layers);
    void setDefaultActivations(Neural::ActivationType hidden, Neural::ActivationType output);

private slots:
    void handleNetworkPathButtonClicked();
    void handleSelectedPathChanged(const QString& text);

private:
    QString defaultLayersText{"784, 10, 10"};
    Neural::ActivationType defaultHiddenActivation{Neural::ActivationType::ReLU};
    Neural::ActivationType defaultOutputActivation{Neural::ActivationType::Softmax};

    QLineEdit* networkPathInput = nullptr;
    QLineEdit* networkLayersInput = nullptr;
    QPushButton* networkPathButton = nullptr;
    QDialogButtonBox* buttonBox = nullptr;

    QComboBox* hiddenActivation = nullptr;
    QComboBox* outputActivation = nullptr;
};
