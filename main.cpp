#include <QApplication>
#include <QSettings>
#include "main_window.h"
#include "digits_recognizer.h"
#include "digits_runner.h"
#include "advanced_terminal.h"


int main(int argc, char** argv) {
    QApplication a(argc, argv);
    a.setApplicationName("Networks");
    a.setApplicationDisplayName("Neural Networks by Kastus");
    a.setOrganizationName("Kastus");

    MainWindow window;
    auto stream = createTerminalOStream(window.getTerminalWidget());

    QSettings settings;
    DigitsRecognizer recognizer;
    DigitsRecognizerController controller(&recognizer);
    controller.loadNetwork(settings.value("last_network_name", QString("network.wgt")).toString());
    controller.setSamplesLimit(50);
    controller.setTestingLimit(150);
    controller.setTrainingDataset(settings.value(
        "last_training_dataset_path",
        QString("/home/kanstancin/Documents/projects/digits-generator/digit_images/")
    ).toString());
    controller.setTestingDataset(settings.value(
        "last_testing_dataset_path",
        QString("/home/kanstancin/Documents/projects/digits-generator/digit_images/")
    ).toString());

    recognizer.setLogger(stream.get());
    recognizer.setSaveOnEachDigit(true);
    window.setLogger(stream.get());
    window.setController(&controller);

    window.show();
    return a.exec();
}
