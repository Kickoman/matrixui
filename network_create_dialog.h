#pragma once

#include <QDialog>


class QLabel;
class QLineEdit;
class QPushButton;
class QDialogButtonBox;


class NetworkCreateDialog : public QDialog
{
    Q_OBJECT
public:
    explicit NetworkCreateDialog(QWidget* parent = nullptr);

    QString getNetworkName() const;
    QVector<unsigned> getLayerSizes() const;

private slots:
    void handleNetworkPathButtonClicked();
    void handleSelectedPathChanged(const QString& text);

private:

    QLineEdit* networkPathInput = nullptr;
    QLineEdit* networkLayersInput = nullptr;
    QPushButton* networkPathButton = nullptr;
    QDialogButtonBox* buttonBox = nullptr;

    // QLabel* networkPathLabel = nullptr;
    // QLabel* networkLayersLabel = nullptr;

};
