#include "gui/lib/network_create_dialog.h"

#include "core/lib/neural_network.h"
#include "core/lib/neural_network_loader.h"

#include "gui/lib/qinputvalidators.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QComboBox>


namespace {

QString LayersToTextRepresentation(const std::vector<std::size_t>& sizes) {
    QStringList layers;
    std::transform(sizes.begin(), sizes.end(), std::back_inserter(layers), [](std::size_t size) {
        return QString::number(size);
    });
    return layers.join(", ");
}

std::vector<std::size_t> TextRepresentationToLayers(const QString& text) {
    const auto sizes = text.split(',', Qt::SkipEmptyParts);
    std::vector<std::size_t> result;
    result.reserve(sizes.size());
    std::transform(sizes.begin(), sizes.end(), std::back_inserter(result), [](const QString& size) {
        return static_cast<std::size_t>(size.toUInt());
    });
    return result;
}

}


NetworkCreateDialog::NetworkCreateDialog(QWidget* parent)
    : QDialog(parent)
{
    auto* mainLayout = new QFormLayout(this);
    auto* networkPathLayout = new QHBoxLayout();

    networkPathInput = new QLineEdit(this);
    networkPathButton = new QPushButton("Select network...", this);
    networkLayersInput = new QLineEdit(this);
    hiddenActivation = new QComboBox(this);
    outputActivation = new QComboBox(this);
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);

    networkLayersInput->setValidator(new CommaSeparatedIntsValidator(networkLayersInput));

    hiddenActivation->addItem("Sigmoid",    static_cast<std::uint8_t>(Neural::ActivationType::Sigmoid));
    hiddenActivation->addItem("ReLU",       static_cast<std::uint8_t>(Neural::ActivationType::ReLU));
    hiddenActivation->addItem("Leaky ReLU", static_cast<std::uint8_t>(Neural::ActivationType::LeakyReLU));
    hiddenActivation->addItem("Tanh",       static_cast<std::uint8_t>(Neural::ActivationType::Tanh));
    hiddenActivation->addItem("SoftMax",    static_cast<std::uint8_t>(Neural::ActivationType::Softmax));
    outputActivation->addItem("Sigmoid",    static_cast<std::uint8_t>(Neural::ActivationType::Sigmoid));
    outputActivation->addItem("ReLU",       static_cast<std::uint8_t>(Neural::ActivationType::ReLU));
    outputActivation->addItem("Leaky ReLU", static_cast<std::uint8_t>(Neural::ActivationType::LeakyReLU));
    outputActivation->addItem("Tanh",       static_cast<std::uint8_t>(Neural::ActivationType::Tanh));
    outputActivation->addItem("SoftMax",    static_cast<std::uint8_t>(Neural::ActivationType::Softmax));

    networkPathLayout->addWidget(networkPathInput);
    networkPathLayout->addWidget(networkPathButton);

    mainLayout->addRow("Network path", networkPathLayout);
    mainLayout->addRow("Layers", networkLayersInput);
    mainLayout->addRow("Hidden layer activation function", hiddenActivation);
    mainLayout->addRow("Output layer activation function", outputActivation);
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

Neural::NeuralNetworkConfiguration NetworkCreateDialog::getConfiguration() const {
    return {
        .hiddenActivation = static_cast<Neural::ActivationType>(hiddenActivation->currentData().toUInt()),
        .outputActivation = static_cast<Neural::ActivationType>(outputActivation->currentData().toUInt()),
        .layersSizes = TextRepresentationToLayers(networkLayersInput->text()),
    };
}

void NetworkCreateDialog::setCurrentNetworkPath(const QString& path) {
    networkPathInput->setText(path);
}

void NetworkCreateDialog::setDefaultLayersText(const QString& layers) {
    defaultLayersText = layers;
}

void NetworkCreateDialog::setDefaultLayersText(const std::vector<std::size_t>& layers) {
    defaultLayersText = LayersToTextRepresentation(layers);
}

void NetworkCreateDialog::setDefaultActivations(Neural::ActivationType hidden, Neural::ActivationType output) {
    defaultHiddenActivation = hidden;
    defaultOutputActivation = output;
}

void NetworkCreateDialog::handleNetworkPathButtonClicked() {
    QFileDialog dialog(this);
    if (const auto path = networkPathInput->text(); QFile::exists(path)) {
        dialog.setDirectory(QFileInfo(path).dir());
    }
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
            const auto config = Neural::LoadConfig(text.toStdString()).value();
            networkLayersInput->setText(LayersToTextRepresentation(config.layersSizes));

            const auto hiddenIndex = hiddenActivation->findData(static_cast<std::uint8_t>(config.hiddenActivation));
            const auto outputIndex = outputActivation->findData(static_cast<std::uint8_t>(config.outputActivation));
            if (hiddenIndex != -1) {
                hiddenActivation->setCurrentIndex(hiddenIndex);
            }
            if (outputIndex != -1) {
                outputActivation->setCurrentIndex(outputIndex);
            }
        } catch (const std::exception& e) {
            qDebug() << "Error: " << e.what();
        }
        networkLayersInput->setEnabled(false);
    } else {
        networkLayersInput->setEnabled(true);
        networkLayersInput->setText(defaultLayersText);
        hiddenActivation->setCurrentIndex(hiddenActivation->findData(static_cast<std::uint8_t>(defaultHiddenActivation)));
        outputActivation->setCurrentIndex(outputActivation->findData(static_cast<std::uint8_t>(defaultOutputActivation)));
    }
}
