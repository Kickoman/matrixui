#include "gui/network_create_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QFileDialog>


NetworkCreateDialog::NetworkCreateDialog(QWidget* parent)
    : QDialog(parent)
{
    auto* mainLayout = new QFormLayout(this);
    auto* networkPathLayout = new QHBoxLayout();

    networkPathInput = new QLineEdit(this);
    networkPathButton = new QPushButton("Select network...", this);
    networkLayersInput = new QLineEdit(this);
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

    networkPathLayout->addWidget(networkPathInput);
    networkPathLayout->addWidget(networkPathButton);

    mainLayout->addRow("Network path", networkPathLayout);
    mainLayout->addRow("Layers", networkLayersInput);
    mainLayout->addRow(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    setWindowTitle("Open network");
    setModal(true);

    connect(networkPathButton, &QPushButton::clicked, this, &NetworkCreateDialog::handleNetworkPathButtonClicked);
    connect(networkPathInput, &QLineEdit::textChanged, this, &NetworkCreateDialog::handleSelectedPathChanged);
}


QString NetworkCreateDialog::getNetworkName() const {
    return networkPathInput->text();
}

QVector<unsigned> NetworkCreateDialog::getLayerSizes() const {
    return {};
}

void NetworkCreateDialog::handleNetworkPathButtonClicked() {
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setNameFilter("*.wgt");
    if (dialog.exec() == QDialog::Accepted) {
        const auto fileNames = dialog.selectedFiles();
        if (fileNames.size() != 1) {
            throw std::runtime_error("Can't open multiple files");
        }
        const auto fileName = fileNames.first();
        networkPathInput->setText(fileName);
    }
}

void NetworkCreateDialog::handleSelectedPathChanged(const QString& text) {
    qDebug() << "Selection: " << text;
    if (QFile::exists(text)) {
        // try loading ???
        networkLayersInput->setEnabled(false);
    } else {
        networkLayersInput->setEnabled(true);
    }
}
