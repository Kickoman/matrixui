#include "main_window.h"
#include "advanced_terminal.h"
#include "digits_recognizer.h"
#include "digits_runner.h"
#include "matrix.h"
#include "time_chart.h"
#include "digit_chart.h"

#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QtCharts/QtCharts>
#include <QTimer>
#include <QRandomGenerator>
#include <qobject.h>
#include <qrandom.h>


MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    terminal = new AdvancedTerminal(this);
    stream = createTerminalOStream(terminal).release();

    chart = new TimeChart(this);
    digitChart = new DigitChart(this);

    auto* vLayout = new QVBoxLayout();
    auto* hLayout = new QHBoxLayout();
    vLayout->addLayout(hLayout);
    vLayout->addWidget(terminal);
    hLayout->addWidget(chart);
    hLayout->addWidget(digitChart);

    auto* mainWidget = new QWidget(this);
    mainWidget->setLayout(vLayout);

    setCentralWidget(mainWidget);

    terminal->write("Hello world");

    // auto* timer = new QTimer(this);
    // connect(timer, &QTimer::timeout, this, &MainWindow::handleTimer);
    // timer->start(2000);
    doDemo();
    showMaximized();

    thread = new QThread(this);
    DigitsRecognizer recognizer;
    recognizer.loadNetwork("interm-6.wgt");
    recognizer.setDataset("/home/kanstancin/Documents/projects/digits-generator/digit_images/");
    recognizer.setLogger(stream);
    runner = new DigitsRunner(recognizer);
    runner->moveToThread(thread);
    connect(thread, &QThread::finished, runner, &QObject::deleteLater);
    connect(thread, &QThread::started, runner, &DigitsRunner::run);
    connect(runner, &DigitsRunner::updatedStatistics, this, &MainWindow::handleStatistics);
    thread->start();
}

MainWindow::~MainWindow()
{
    runner->requestStop();
    thread->quit();
    thread->wait();
    delete stream;
}

void MainWindow::handleTimer()
{
    const double value = QRandomGenerator::global()->bounded(100.0);
    logger() << "Appending " << value << std::endl;
    chart->addPoint(value);

    for (unsigned i = 0; i < 10; ++i) {
        const double val = QRandomGenerator::global()->bounded(5.0);
        digitChart->setValue(i, val);
        logger() << "Set " << val << " for " << i << std::endl;
    }
}

void MainWindow::handleStatistics(const TestResult& result) {
    const auto& total = result.getTotal();
    const double rate = total.totalTests > 0 ? 100.0 * total.passedTests / total.totalTests : 0;
    chart->addPoint(rate);

    for (unsigned i = 0; i < 10; ++i) {
        const auto& res = result.digits[i];
        const double rate = res->totalTests > 0 ? 100.0 * res->passedTests / res->totalTests : 0;
        digitChart->setValue(i, rate);
    }
}

std::ostream& MainWindow::logger() {
    return *stream;
}

void MainWindow::doDemo()
{
    using namespace std::chrono;
    logger() << "=== Matrix Multiplication Library Demonstration ===\n\n";

    // Create sample matrices
    logger() << "Creating matrices A (3x2) and B (2x3):\n";
    Matrix A(3, 2);
    A(0, 0) = 1.0; A(0, 1) = 2.0;
    A(1, 0) = 3.0; A(1, 1) = 4.0;
    A(2, 0) = 5.0; A(2, 1) = 6.0;

    Matrix B(2, 3);
    B(0, 0) = 7.0; B(0, 1) = 8.0; B(0, 2) = 9.0;
    B(1, 0) = 10.0; B(1, 1) = 11.0; B(1, 2) = 12.0;

    logger() << "Matrix A:\n";
    A.print();
    logger() << "\nMatrix B:\n";
    B.print();

    // Demonstrate naive multiplication
    logger() << "\n=== Naive Matrix Multiplication ===\n";
    auto start_naive = high_resolution_clock::now();
    Matrix C_naive = A * B;
    auto stop_naive = high_resolution_clock::now();

    logger() << "Result (A * B):\n";
    C_naive.print();

    auto duration_naive = duration_cast<microseconds>(stop_naive - start_naive);
    logger() << "Execution time: " << duration_naive.count() << " microseconds\n";

    // Demonstrate optimized multiplication
    logger() << "\n=== Optimized Matrix Multiplication ===\n";
    auto start_optimized = high_resolution_clock::now();
    Matrix C_optimized = A.multiplyOptimized(B);
    auto stop_optimized = high_resolution_clock::now();

    logger() << "Result (A.multiply_optimized(B)):\n";
    C_optimized.print();

    auto duration_optimized = duration_cast<microseconds>(stop_optimized - start_optimized);
    logger() << "Execution time: " << duration_optimized.count() << " microseconds\n";

    // Verify results are the same
    logger() << "\n=== Verification ===\n";
    bool results_match = true;
    for (size_t i = 0; i < C_naive.getRows(); ++i) {
        for (size_t j = 0; j < C_naive.getCols(); ++j) {
            if (std::abs(C_naive(i, j) - C_optimized(i, j)) > 1e-9) {
                results_match = false;
                break;
            }
        }
    }

    logger() << "Results match: " << (results_match ? "YES" : "NO") << std::endl;

    // Performance comparison
    logger() << "\n=== Performance Comparison ===\n";
    logger() << "Naive multiplication time: " << duration_naive.count() << " μs\n";
    logger() << "Optimized multiplication time: " << duration_optimized.count() << " μs\n";
    logger() << "Speedup: " << (double)duration_naive.count() / duration_optimized.count() << "x\n";

    // Demonstrate with larger matrices for better performance comparison
    logger() << "\n=== Large Matrix Example (100x100) ===\n";

    // Create larger matrices
    const size_t large_size = 100;
    Matrix large_A = Matrix::ones(large_size, large_size) * 2.0;
    Matrix large_B = Matrix::identity(large_size) * 3.0;

    logger() << "Created two matrices " << large_size << "x" << large_size << "\n";

    // Time optimized multiplication with larger matrices
    auto start_large = high_resolution_clock::now();
    Matrix large_result = large_A.multiplyOptimized(large_B);
    auto stop_large = high_resolution_clock::now();

    auto duration_large = duration_cast<milliseconds>(stop_large - start_large);
    logger() << "Time for optimized multiplication of " << large_size << "x" << large_size
         << " matrices: " << duration_large.count() << " milliseconds\n";

    // Verify the result (should be 6.0 for all elements)
    bool correct_result = true;
    for (size_t i = 0; i < large_size; ++i) {
        for (size_t j = 0; j < large_size; ++j) {
            double expected = 6.0;
            if (std::abs(large_result(i, j) - expected) > 1e-9) {
                correct_result = false;
                break;
            }
        }
    }

    logger() << "Large matrix result correct: " << (correct_result ? "YES" : "NO") << std::endl;
}
