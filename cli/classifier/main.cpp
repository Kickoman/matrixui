#include "cli/classifier/commands.h"
#include "cli/classifier/options.h"

#include "cli/lib/argv_scan.h"
#include "cli/lib/json_config.h"

#include "core/nn/layers.h"

#include <CLI11/CLI11.hpp>

#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv) {
    CLI::App app{"MatrixGui headless classifier CLI"};
    app.require_subcommand(1);

    // Assigned by whichever subcommand callback runs; require_subcommand(1) makes
    // that exactly one.
    int exitCode = ClassifierCli::kSuccess;

    ClassifierCli::PredictOptions predict;
    CLI::App* predictCmd = app.add_subcommand("predict", "Classify a single PNG image using a trained network");

    predictCmd->add_option("--network", predict.networkPath, "Trained network file (.wgt) to load")
        ->required()
        ->check(CLI::ExistingFile);
    predictCmd->add_option("--image", predict.imagePath, "PNG image to classify")
        ->required()
        ->check(CLI::ExistingFile);
    predictCmd->add_option("--dataset-img-width", predict.imageWidth, "Width of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();
    predictCmd->add_option("--dataset-img-height", predict.imageHeight, "Height of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();
    predictCmd->callback([&] {
        exitCode = ClassifierCli::Predict(std::cout, std::cerr, predict);
    });


    CLI::App* trainCmd = app.add_subcommand("train", "Train (or continue training) a classifier network on a directory dataset");

    ClassifierCli::TrainOptions train;

    // Load-bearing ordering: do not move this block below the registrations. The
    // config files are read straight out of argv, before the per-field options
    // exist, so that a config file supplies the new defaults and a per-field flag
    // still overrides just that field -- and so the capture_default_str() calls
    // below snapshot the loaded values, making --help show what will be used.
    train.learningConfigPath = CliLib::FindOptionValue(argc, argv, "--learning-config");
    train.networkConfigPath = CliLib::FindOptionValue(argc, argv, "--network-config");
    if (!CliLib::LoadJsonConfig(std::cerr, train.learningConfigPath, train.learning, "learning")) {
        return ClassifierCli::kBadConfig;
    }
    if (!CliLib::LoadJsonConfig(std::cerr, train.networkConfigPath, train.network, "network")) {
        return ClassifierCli::kBadConfig;
    }

    trainCmd->add_option("--network", train.networkPath, "Network save path (.wgt). Created if missing.")
        ->required();

    trainCmd->add_option("--working-directory", train.workingDirectory, "Working directory for training data.")
        ->capture_default_str();

    trainCmd->add_option("--learning-config", train.learningConfigPath,
            "Load training hyperparameters from a JSON file (as written to learning-config.json); "
            "flags below override individual fields from the loaded config.")
        ->group("Config")
        ->check(CLI::ExistingFile);
    trainCmd->add_option("--network-config", train.networkConfigPath,
            "Load network topology from a JSON file (as written to network-config.json), used only "
            "when creating a new network; flags below override individual fields from the loaded config.")
        ->group("Config")
        ->check(CLI::ExistingFile);

    trainCmd->add_option("--dataset", train.datasetPath,
            "Dataset root used for both training and testing (subdirs 0..9). "
            "Overridden per-side by --train-dataset/--test-dataset.")
        ->group("Dataset");
    trainCmd->add_option("--train-dataset", train.trainDatasetPath,
            "Training data root (subdirs 0..<num_of_classes-1>); falls back to --dataset if not set.")
        ->group("Dataset");
    trainCmd->add_option("--test-dataset", train.testDatasetPath,
            "Testing data root (subdirs 0..<num_of_classes-1>); falls back to --dataset if not set.")
        ->group("Dataset");
    trainCmd->add_option("--test-file-limit", train.testFileLimit,
            "Max test files per class (0 = all). Runs evaluation after training.")
        ->group("Dataset")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-width", train.imageWidth, "Width of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-height", train.imageHeight, "Height of test images in pixels.")
        ->group("Dataset")
        ->capture_default_str();

    trainCmd->add_option("--layers", train.network.layersSizes,
            "Layer sizes, comma-separated (used only when creating a new network)")
        ->delimiter(',')
        ->group("Network")
        ->capture_default_str();

    // Deliberately narrower than the NLOHMANN_JSON_SERIALIZE_ENUM in
    // core/nn/layers.h: --hidden-activation leakyrelu stays rejected.
    const std::map<std::string, Neural::ActivationType> activationMap{
        {"sigmoid", Neural::ActivationType::Sigmoid}, {"relu", Neural::ActivationType::ReLU},
        {"tanh", Neural::ActivationType::Tanh}, {"softmax", Neural::ActivationType::Softmax}};

    trainCmd->add_option("--hidden-activation", train.network.hiddenActivation,
            "Hidden layer activation (used only when creating a new network)")
        ->transform(CLI::CheckedTransformer(activationMap, CLI::ignore_case))
        ->group("Network")
        ->capture_default_str();
    trainCmd->add_option("--output-activation", train.network.outputActivation,
            "Output layer activation (used only when creating a new network)")
        ->transform(CLI::CheckedTransformer(activationMap, CLI::ignore_case))
        ->group("Network")
        ->capture_default_str();

    trainCmd->add_option("--initial-lr", train.learning.initialLearningRate, "Initial learning rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--min-lr", train.learning.minLearningRate, "Minimum learning rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--lr-decay", train.learning.learningRateDecay, "Learning rate decay factor")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--max-epochs", train.learning.maxEpochs, "Maximum training epochs")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--patience", train.learning.patience, "Early-stopping patience (epochs)")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--inner-epochs", train.learning.innerEpochs, "Backprop passes per sample per epoch")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--dropout", train.learning.dropoutRate, "Dropout rate")
        ->group("Training")->capture_default_str();
    trainCmd->add_option("--dataset-limit-per-label,--dataset-file-limit", train.learning.datasetLimitPerLabel,
            "Max training files per class (0 = all)")
        ->group("Training")->capture_default_str();

    trainCmd->callback([&] {
        exitCode = ClassifierCli::Train(std::cout, std::cerr, train);
    });

    // CLI11_PARSE cannot be used here: the subcommand callbacks run inside parse(),
    // and the macro's catch covers only CLI::ParseError.
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& error) {
        return app.exit(error);
    }

    return exitCode;
}
