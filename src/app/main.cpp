#include <QApplication>
#include "gui/main_window.h"
#include "core/digits_recognizer.h"
#include "gui/digits_runner.h"
#include "gui/advanced_terminal.h"


int main(int argc, char** argv) {
    QApplication a(argc, argv);

    MainWindow window;
    auto stream = createTerminalOStream(window.getTerminalWidget());

    DigitsRecognizer recognizer;
    DigitsRecognizerController controller(&recognizer);
    controller.loadNetwork("interm-6.wgt");
    controller.setSamplesLimit(10);
    controller.setTestingLimit(150);
    controller.setDataset("/home/kanstancin/Documents/projects/digits-generator/digit_images/");

    recognizer.setLogger(stream.get());
    window.setLogger(stream.get());
    window.setController(&controller);

    window.show();
    return a.exec();
}
