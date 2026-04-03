#include <QApplication>
#include <QSettings>
#include "main_window.h"
#include "trainer.h"
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
    Neural::Trainer recognizer;
    DigitsRecognizerController controller(&recognizer);
    controller.loadNetwork(settings.value("last_network_name", QString("network.wgt")).toString());
    controller.setTrainingDataset(settings.value(
        "last_training_dataset_path",
        QString("/home/kanstancin/Documents/projects/digits-generator/digit_images/")
    ).toString());
    controller.setTestingDataset(settings.value(
        "last_testing_dataset_path",
        QString("/home/kanstancin/Documents/projects/digits-generator/digit_images/")
    ).toString());

    recognizer.setOutputStream(stream.get());
    window.setLogger(stream.get());
    window.setController(&controller);

    window.show();
    return a.exec();
}
