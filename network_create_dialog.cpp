#include "network_create_dialog.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QFileDialog>

#include <fstream>
#include <qfiledialog.h>
#include <qnamespace.h>

#include "qinputvalidators.h"


NetworkCreateDialog::NetworkCreateDialog(QWidget* parent)
    : QDialog(parent)
{
    auto* mainLayout = new QFormLayout(this);
    auto* networkPathLayout = new QHBoxLayout();

    networkPathInput = new QLineEdit(this);
    networkPathButton = new QPushButton("Select network...", this);
    networkLayersInput = new QLineEdit(this);
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

    networkLayersInput->setValidator(new CommaSeparatedIntsValidator(networkLayersInput));

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
    const auto sizes = networkLayersInput->text().split(',', Qt::SkipEmptyParts);
    QVector<unsigned> result;
    result.reserve(sizes.size());
    std::transform(sizes.begin(), sizes.end(), std::back_inserter(result), [](const QString& size) {
        return size.toUInt();
    });
    return result;
}

void NetworkCreateDialog::handleNetworkPathButtonClicked() {
    QFileDialog dialog(this);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setOption(QFileDialog::DontConfirmOverwrite, true);
    dialog.setNameFilter("*.wgt");
    if (dialog.exec() == QDialog::Accepted) {
        const auto fileNames = dialog.selectedFiles();
        if (fileNames.size() != 1) {
            throw std::runtime_error("Can't open multiple files");
        }
        const auto fileName = fileNames.first();
        networkPathInput->setText(fileName + (fileName.endsWith(".wgt") ? "" : ".wgt"));
    }
}

void NetworkCreateDialog::handleSelectedPathChanged(const QString& text) {
    qDebug() << "Selection: " << text;
    if (QFile::exists(text)) {
        try {
            std::size_t numLayers;
            std::ifstream in(text.toStdString());
            in >> numLayers;
            std::vector<std::size_t> layerSizes(numLayers);
            for (std::size_t i = 0; i < numLayers; ++i) {
                in >> layerSizes[i];
            }
            QStringList layers;
            std::transform(layerSizes.begin(), layerSizes.end(), std::back_inserter(layers), [](std::size_t size) {
                return QString::number(size);
            });
            networkLayersInput->setText(layers.join(", "));
        } catch (const std::exception& e) {
            qDebug() << "Error: " << e.what();
        }
        networkLayersInput->setEnabled(false);
    } else {
        networkLayersInput->setEnabled(true);
        networkLayersInput->setText("");
    }
}
