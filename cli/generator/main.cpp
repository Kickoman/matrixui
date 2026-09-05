#include "cli/generator/commands.h"
#include "cli/generator/options.h"

#include <CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"MatrixGui GAN CLI (generator/discriminator training & sampling)"};
    app.require_subcommand(1);

    int exitCode = GeneratorCli::kSuccess;

    GeneratorCli::GenerateOptions generate;
    CLI::App* generateCmd = app.add_subcommand("generate", "Generate sample images from a trained generator");

    generateCmd->add_option("--label", generate.label, "Digit to generate")->required();
    generateCmd->add_option("--num-classes", generate.numClasses, "Number of digit classes")->capture_default_str();
    generateCmd->add_option("--generator", generate.generatorPath, "Generator network file to load")
        ->check(CLI::ExistingFile)
        ->capture_default_str();
    generateCmd->add_option("--num-samples", generate.numSamples, "How many images to generate")->capture_default_str();
    generateCmd->add_option("--output", generate.outputPath,
            "Output PNG path. With --num-samples > 1, a counter is inserted before the extension.")
        ->capture_default_str();
    generateCmd->add_option("--classifier", generate.classifierPath,
            "Optional: print this classifier's predicted label alongside each image")
        ->check(CLI::ExistingFile);
    generateCmd->add_option("--dataset-img-width", generate.imageWidth, "Width of test images in pixels.")
        ->capture_default_str();
    generateCmd->add_option("--dataset-img-height", generate.imageHeight, "Height of test images in pixels.")
        ->capture_default_str();
    generateCmd->callback([&] {
        exitCode = GeneratorCli::Generate(std::cout, std::cerr, generate);
    });

    GeneratorCli::TrainOptions train;
    CLI::App* trainCmd = app.add_subcommand("train", "Train generator+discriminator against a frozen classifier");

    trainCmd->add_option("--classifier", train.classifierPath, "Trained classifier (frozen signal source)")->required();
    trainCmd->add_option("--dataset", train.datasetPath, "Training images root (subdirs 0..9 of PNGs)")
        ->required()
        ->check(CLI::ExistingDirectory);

    trainCmd->add_option("--generator", train.generatorPath, "Generator save path (created if missing)")
        ->group("Network paths")->capture_default_str();
    trainCmd->add_option("--discriminator", train.discriminatorPath, "Discriminator save path (created if missing)")
        ->group("Network paths")->capture_default_str();

    trainCmd->add_option("--latent-dim", train.config.latentDim, "Noise vector size fed to generator")
        ->group("Topology")->capture_default_str();
    trainCmd->add_option("--gen-layers", train.generatorHidden, "Generator hidden layer sizes (new network only)")
        ->delimiter(',')->group("Topology")->capture_default_str();
    trainCmd->add_option("--disc-layers", train.discriminatorHidden, "Discriminator hidden layer sizes (new network only)")
        ->delimiter(',')->group("Topology")->capture_default_str();
    trainCmd->add_option("--dataset-img-width", train.imageWidth, "Width of test images in pixels.")
        ->group("Topology")
        ->capture_default_str();
    trainCmd->add_option("--dataset-img-height", train.imageHeight, "Height of test images in pixels.")
        ->group("Topology")
        ->capture_default_str();

    trainCmd->add_option("--gen-lr", train.config.generatorLearningRate, "Generator learning rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--disc-lr", train.config.discriminatorLearningRate, "Discriminator learning rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--epochs", train.config.epochs, "Total epochs")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--batch-size", train.config.batchSize, "Samples per batch")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--disc-steps", train.config.discriminatorStepsPerGenStep,
            "Discriminator updates per generator update")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--dropout", train.config.dropoutRate, "Discriminator dropout rate")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--classifier-weight", train.config.classifierLossWeight,
            "Weight of classifier loss in generator update")
        ->group("GAN training")->capture_default_str();
    trainCmd->add_option("--dataset-limit", train.config.datasetLimitPerLabel, "Max images per label to load (0 = all)")
        ->group("GAN training")->capture_default_str();

    trainCmd->add_flag("--adaptive-lr", train.config.adaptiveLr.enabled, "Enable adaptive lr adjustment")
        ->group("Adaptive learning rate");
    trainCmd->add_option("--lr-ema-alpha", train.config.adaptiveLr.lrEmaAlpha, "EMA smoothing factor; higher = slower reaction")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--d-real-target-low", train.config.adaptiveLr.dRealTargetLow, "D(real) below this => D collapsed")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--d-real-target-high", train.config.adaptiveLr.dRealTargetHigh, "D(real) above this => D dominating")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--gen-fool-target-low", train.config.adaptiveLr.dFakeTargetLow,
            "D(G(z)) below this => D dominating/collapsed")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--gen-fool-target-high", train.config.adaptiveLr.dFakeTargetHigh, "D(G(z)) above this => G dominating")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-adjust-factor", train.config.adaptiveLr.lrAdjustFactor, "Multiplicative lr step per epoch")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-min", train.config.adaptiveLr.lrMin, "Lower lr clamp")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-max", train.config.adaptiveLr.lrMax, "Upper lr clamp")
        ->group("Adaptive learning rate")->capture_default_str();
    trainCmd->add_option("--lr-warmup-epochs", train.config.adaptiveLr.lrWarmupEpochs, "Epochs before adaptive adjustments begin")
        ->group("Adaptive learning rate")->capture_default_str();

    trainCmd->add_flag("--flatness-detection", train.config.flatnessDetection.enabled, "Enable plateau detection")
        ->group("Flatness detection");
    trainCmd->add_option("--flatness-threshold", train.config.flatnessDetection.threshold,
            "Max EMA change per epoch to count as flat")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-window", train.config.flatnessDetection.window, "Consecutive flat epochs before kick fires")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-kick-duration", train.config.flatnessDetection.kickDuration, "Epochs to hold the kick")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-dropout-boost", train.config.flatnessDetection.discriminatorDropoutBoost,
            "Multiply D dropout by this during kick")
        ->group("Flatness detection")->capture_default_str();
    trainCmd->add_option("--flatness-gen-lr-boost", train.config.flatnessDetection.generatorLrBoost, "Multiply G lr by this during kick")
        ->group("Flatness detection")->capture_default_str();

    trainCmd->callback([&] {
        exitCode = GeneratorCli::Train(std::cout, std::cerr, train);
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
