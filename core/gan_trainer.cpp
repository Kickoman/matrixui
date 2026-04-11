#include "gan_trainer.h"
#include "loss_functions.h"
#include <algorithm>
#include <numeric>
#include <random>

namespace Neural {

GanTrainer::GanTrainer(
    Generator generator,
    Discriminator discriminator,
    NeuralNetworkApplier classifier
)
    : generator(std::move(generator))
    , discriminator(std::move(discriminator))
    , classifier(std::move(classifier))
{}

void GanTrainer::trainDiscriminatorStep(
    const Matrix& realSample,
    const Matrix& fakeSample,
    const double lr,
    const double dropoutRate
) {
    discriminator.zeroGradients();

    // Real sample: target is 1 (real).
    const Matrix realScore = discriminator.forward(realSample, dropoutRate);
    discriminator.backward(bce_gradient(realScore, Matrix(1, 1, 1.0)));

    // Fake sample: target is 0 (fake).
    // Gradients accumulate on top of the real-sample pass above.
    const Matrix fakeScore = discriminator.forward(fakeSample, dropoutRate);
    discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 0.0)));

    discriminator.applyGradients(lr);
}

double GanTrainer::trainGeneratorStep(
    const std::size_t label,
    const double classifierLossWeight,
    const double lr
) {
    generator.zeroGradients();
    // Zero D and C gradients so their backward passes give clean input-space
    // gradients. Their accumulated weight gradients are discarded: D's are
    // overwritten at the start of the next discriminator step; C's are never
    // applied at all.
    discriminator.zeroGradients();
    classifier.zeroGradients();

    const Matrix fake = generator.generate(label);

    // --- Discriminator loss: fool D into scoring fake as real ---
    const Matrix fakeScore = discriminator.forward(fake, 0.0);
    const Matrix inputGradFromD = discriminator.backward(bce_gradient(fakeScore, Matrix(1, 1, 1.0)));

    // --- Classifier loss: fake should be classified as 'label' ---
    const Matrix classifierOut = classifier.forward(fake, 0.0);
    // Build one-hot target for the requested label.
    Matrix oneHot(1, classifierOut.getCols(), 0.0);
    oneHot(0, label) = 1.0;
    // SoftmaxLayer::backward is identity (assumes CE loss upstream), so
    // (classifierOut - oneHot) is the correct gradient at the classifier input.
    const Matrix inputGradFromC = classifier.backward(classifierOut - oneHot);

    // Combine: both gradients point toward improving the generator.
    const Matrix combinedGrad = inputGradFromD + inputGradFromC * classifierLossWeight;

    generator.backward(combinedGrad);
    generator.applyGradients(lr);

    return fakeScore(0, 0);
}

void GanTrainer::train(
    const std::vector<Matrix>& realSamples,
    const GanConfig& config,
    std::ostream* log
) {
    if (realSamples.empty()) return;

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> labelDist(0, config.numClasses - 1);
    std::vector<std::size_t> indices(realSamples.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double totalDiscScore = 0.0;
        double totalGenScore = 0.0;
        std::size_t steps = 0;

        for (std::size_t i = 0; i + config.batchSize <= indices.size(); i += config.batchSize) {
            // Pick the label for this step — used for both fake generation in D
            // training and for the conditioned G update.
            const std::size_t label = labelDist(rng);

            // Train discriminator for discriminatorStepsPerGenStep steps.
            for (std::size_t d = 0; d < config.discriminatorStepsPerGenStep; ++d) {
                const std::size_t idx = indices[i + (d % config.batchSize)];
                const Matrix fake = generator.generate(label);
                trainDiscriminatorStep(
                    realSamples[idx], fake,
                    config.discriminatorLr, config.dropoutRate
                );
                totalDiscScore += discriminator.score(realSamples[idx])(0, 0);
            }

            // Train generator one step conditioned on the chosen label.
            totalGenScore += trainGeneratorStep(
                label, config.classifierLossWeight, config.generatorLr
            );
            ++steps;
        }

        if (log && steps > 0) {
            const double avgDiscReal = totalDiscScore / (steps * config.discriminatorStepsPerGenStep);
            const double avgGenFool  = totalGenScore / steps;

            // Sample a generated image for each class and log what the classifier thinks.
            const std::size_t logLabel = epoch % config.numClasses;
            const Matrix sample = generator.generate(logLabel);
            const Matrix classifierOut = classifier.predict(sample);
            std::size_t predictedLabel = 0;
            for (std::size_t c = 1; c < classifierOut.getCols(); ++c)
                if (classifierOut(0, c) > classifierOut(0, predictedLabel))
                    predictedLabel = c;

            *log << "Epoch " << epoch + 1 << "/" << config.epochs
                 << "  D(real)=" << avgDiscReal
                 << "  D(G(z))=" << avgGenFool
                 << "  sample(label=" << logLabel << " -> classifier=" << predictedLabel << ")"
                 << "\n";
        }
    }
}

const Generator& GanTrainer::getGenerator() const { return generator; }
const Discriminator& GanTrainer::getDiscriminator() const { return discriminator; }

}
