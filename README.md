# MatrixGui

Neural network tooling for handwritten digit recognition and generation. The project ships a Qt-based GUI and headless CLI tools for training and inference without a display.

## Modes

| Mode | GUI tab | CLI binary | Description |
|------|---------|------------|-------------|
| Classifier | Classifier | `MatrixGui_headless` | Train a network to classify digits 0–9 |
| Recognizer | Recognizer | `MatrixGui_headless --predict-image` | Draw or load a digit and classify it |
| Generator (GAN) | Generator | `MatrixGui_gan` | Train a conditional GAN and generate synthetic digit images |

## Quick start

**Build everything:**

```bash
cmake -B build
cmake --build build
```

**Train a classifier:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Classify a single image:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --predict-image samples/seven.png
```

**Train a GAN** (requires a trained classifier):

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Generate digit images:**

```bash
./build/MatrixGui_gan \
  --generate --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

## Dataset layout

Training and testing expect a root directory whose immediate subdirectories are named `0` through `9`, each containing PNG images of that digit class:

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Images are loaded at 28×28 pixels and flattened to a 784-element input vector.

## Documentation

- [Building](docs/building.md) — CMake options, build targets, dependencies
- [Word embeddings](docs/words.md) — SGNS pipeline, module layout, CLI reference
- [Classifier](docs/classifier.md) — Training and inference CLI reference
- [GAN](docs/gan.md) — GAN training and image generation CLI reference
- [Hyperparameters](docs/hyperparameters.md) — Learning rate schedules, architecture guidance, GAN stability tricks

---

# MatrixGUI

**Heta nie production rašeńnie!**

Heta prajekt pa vyvučeńni pryncypaŭ raboty neŭronnych sietak. Sproba stvaryć z nulia framework dlia stvareńnia roznaha kštaltu madeliaŭ, trenavańnia i inferensu.
Tut jość nabor klasaŭ dlia vykarystańńia ŭ svaich pragramach, jość versii pragramy trenavańnia i inferensu z kamandnaha radku, versii z grafičnym interfejsam.

## Režymy

| Režym | Kartka interfejsu | Binarnik CLI | Apisańnie |
|------|---------|------------|-------------|
| Klasifikatar | Classifier | `MatrixGui_headless` | Navučaje madeĺ dlia klasifikacyi ličbaŭ ad 1 da 9 |
| Raspaznavaĺnik | Recognizer | `MatrixGui_headless --predict-image` | Klasifikuje zadadzienuju vyjavu (z fajlu) |
| Generatar GAN | Generator | `MatrixGui_gan` | Navučaje cGAN-madeĺ i generuje syntetyčnyja vyjavy ličbaŭ |

## Chutki start

**Zborka ŭsiaho prajektu:**

```bash
cmake -B build
cmake --build build
```

**Navučyć ulasny klasifikatar:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \  # vaš šliach da novaj madeli
  --dataset /path/to/dataset
```

Pra farmat datasetu hliadzi nižej.

**Klasifikavać (raspaznać) vyjavu:**

```bash
./build/MatrixGui_headless \
  --network models/digits.wgt \
  --predict-image samples/seven.png
```

**Navučyć generatar (GAN)**:

Patrabuje navučany klasifikatar!

```bash
./build/MatrixGui_gan \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Zgeneravać vyjavu ličby:**

```bash
./build/MatrixGui_gan \
  --generate --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

## Farmat datasetu

Navučaĺny i testavy dataset musić być dyrektoryjaj, čyje niepasrednyja pad-dyrektoryi nazvanyja ad `0` da `9`, kožnaja ź jakich utrymoŭvaje PNG-vyjavy adpaviednaj ličby:

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Vyjavy čakajucca ŭ pamiery 28x28 pikseliaŭ, jakija potym transfarmujucca va ŭvachodny adnamierny vektar pamieram u 784 elementy.

## Dakumentacyja (pa-angieĺsku)

- [Zborka](docs/building.md) — opcyi CMake, mety zborki, zaliežnaści
- [Klasifikatar](docs/classifier.md) — apisańnie navučańnia i inferensu ŭtylitaj kamandnaha radku
- [GAN](docs/gan.md) — navučańnie GAN i generacyja vyjaŭ utylitaj kamandnaha radku
- [Hiperparametry](docs/hyperparameters.md) — rasklad uzroŭniu navyčańnia (learning rate), dapamožnik pa architektury, parady dlia stabilizacyi GAN
