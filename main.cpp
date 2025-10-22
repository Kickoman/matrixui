#include <QApplication>
#include "main_window.h"
#include "digits_recognizer.h"
#include "digits_runner.h"
#include "advanced_terminal.h"


int main(int argc, char** argv) {
    QApplication a(argc, argv);

    MainWindow window;
    auto stream = createTerminalOStream(window.getTerminalWidget());

    DigitsRecognizer recognizer;
    DigitsRecognizerController controller(&recognizer);
    controller.loadNetwork("interm-6.wgt");
    controller.setSamplesLimit(30);
    controller.setDataset("/home/kanstancin/Documents/projects/digits-generator/digit_images/");

    recognizer.setLogger(stream.get());
    window.setLogger(stream.get());
    window.setController(&controller);

    window.show();
    return a.exec();
}
