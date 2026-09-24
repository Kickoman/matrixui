# MatrixGui

Neural network tooling built from scratch, for learning how neural networks
actually work. It covers handwritten digit recognition and generation plus word
embeddings, and ships both a Qt desktop application and headless CLI tools that
train and run inference without a display.

**This is not a production solution.** It is a study project: an attempt to build
a framework for models, training and inference from nothing.

## Modes

| Mode | CLI binary | Description |
|------|------------|-------------|
| Classifier | `MatrixGui_headless train` | Train a network to classify images into N labelled classes (digits 0–9 in the shipped examples) |
| Recognizer | `MatrixGui_headless predict` | Draw or load an image and classify it |
| Generator (GAN) | `MatrixGui_gan` | Train a conditional GAN and generate synthetic digit images |
| Words | `MatrixGui_words` | Train SGNS word embeddings and explore them (neighbours, analogies, evaluation) |
| Functions | `MatrixGui_functions` | Genetic symbolic regression: evolve an arithmetic expression fitting a table of data points |

Plus one tool that is not a mode. `MatrixGui_models list --root <dir>` walks a
directory of model artifacts and prints what loads, and why the rest is refused;
`MatrixGui_models serve --root <dir>` holds the same composition in a registry and
answers predictions over HTTP, reloadable without dropping a request. The model
side is [`core/serving/`](core/serving/), the HTTP side
[`cli/serving/`](cli/serving/).

All five are also available as modes in the GUI.

## Quick start

**Build everything:**

```bash
git submodule update --init
cmake -B build
cmake --build build
```

**Train a classifier:**

```bash
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Classify a single image:**

```bash
./build/MatrixGui_headless predict \
  --network models/digits.wgt \
  --image samples/seven.png
```

**Train a GAN** (requires a trained classifier):

```bash
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Generate digit images:**

```bash
./build/MatrixGui_gan generate \
  --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

**Train word embeddings:**

```bash
./build/MatrixGui_words buildvoc --input-file corpus.txt --output-file built.voc
./build/MatrixGui_words buildcor --input-file corpus.txt --vocabulary built.voc --output-file built.cor
./build/MatrixGui_words train --vocabulary built.voc --corpus built.cor --output-file emb.bin
./build/MatrixGui_words neighbours --vocabulary built.voc --embeddings emb.bin --word king
```

## Dataset layout

Training and testing expect a root directory whose immediate subdirectories are
named `0` through `N-1`, where `N` is the network's output size, each containing
PNG images of that class:

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Images are read at 28×28 pixels by default and flattened into a single input
vector of 784 elements. Both dimensions are configurable with
`--dataset-img-width` and `--dataset-img-height`, as long as `width × height`
matches the network's input layer.

## Documentation

- [Building](docs/building.md) — CMake options, build targets, dependencies, tests
- [Classifier](docs/classifier.md) — Training and inference CLI reference
- [GAN](docs/gan.md) — GAN training and image generation CLI reference
- [Word embeddings](docs/words.md) — SGNS pipeline, module layout, CLI reference
- [Hyperparameters](docs/hyperparameters.md) — Learning rate schedules, architecture guidance, GAN stability tricks
- [GUI](docs/gui.md) — The desktop application: modes, threading, the shared terminal

---

<details>
<summary><b>Pa-biełarusku (Łacinka)</b></summary>

# MatrixGUI

**Heta nie production rašeńnie!**

Heta prajekt pa vyvučeńni pryncypaŭ raboty neŭronnych sietak. Sproba stvaryć z nulia framework dlia stvareńnia roznaha kštaltu madeliaŭ, trenavańnia i inferensu.
Tut jość nabor klasaŭ dlia vykarystańńia ŭ svaich pragramach, jość versii pragramy trenavańnia i inferensu z kamandnaha radku, versii z grafičnym interfejsam.

## Režymy

| Režym | Binarnik CLI | Apisańnie |
|-------|--------------|-----------|
| Klasifikatar | `MatrixGui_headless train` | Navučaje madeĺ dlia klasifikacyi vyjavaŭ pa N klasach (ličby ad 0 da 9 u prykładach) |
| Raspaznavaĺnik | `MatrixGui_headless predict` | Klasifikuje zadadzienuju vyjavu (z fajlu) |
| Generatar GAN | `MatrixGui_gan` | Navučaje cGAN-madeĺ i generuje syntetyčnyja vyjavy ličbaŭ |
| Słovy | `MatrixGui_words` | Navučaje viektarnyja pradstaŭleńni słovaŭ (SGNS) i dazvalaje ich dasledavać |
| Funkcyi | `MatrixGui_functions` | Genetyčnaja simvaĺnaja regresija: evalucyja arytmetyčnaha vyrazu pad zadadzienyja punkty |

Usie piać režymaŭ dastupnyja taksama ŭ grafičnym interfejsie.

## Chutki start

**Zborka ŭsiaho prajektu:**

```bash
git submodule update --init
cmake -B build
cmake --build build
```

**Navučyć ulasny klasifikatar:**

```bash
./build/MatrixGui_headless train \
  --network models/digits.wgt \
  --dataset /path/to/dataset
```

Dzie `--network` — heta vaš šliach da novaj madeli. Pra farmat datasetu hliadzi nižej.

**Klasifikavać (raspaznać) vyjavu:**

```bash
./build/MatrixGui_headless predict \
  --network models/digits.wgt \
  --image samples/seven.png
```

**Navučyć generatar (GAN)**:

Patrabuje navučany klasifikatar!

```bash
./build/MatrixGui_gan train \
  --classifier models/digits.wgt \
  --dataset /path/to/mnist_split
```

**Zgeneravać vyjavu ličby:**

```bash
./build/MatrixGui_gan generate \
  --label 7 \
  --generator models/generator.wgt \
  --output samples/seven.png
```

**Navučyć viektarnyja pradstaŭleńni słovaŭ:**

```bash
./build/MatrixGui_words buildvoc --input-file corpus.txt --output-file built.voc
./build/MatrixGui_words buildcor --input-file corpus.txt --vocabulary built.voc --output-file built.cor
./build/MatrixGui_words train --vocabulary built.voc --corpus built.cor --output-file emb.bin
./build/MatrixGui_words neighbours --vocabulary built.voc --embeddings emb.bin --word king
```

## Farmat datasetu

Navučaĺny i testavy dataset musić być dyrektoryjaj, čyje niepasrednyja pad-dyrektoryi nazvanyja ad `0` da `N-1`, dzie `N` — pamier vyjściovaha słoja sietki. Kožnaja ź ich utrymoŭvaje PNG-vyjavy adpaviednaha kłasu:

```
data/mnist_split/
  0/  *.png
  1/  *.png
  ...
  9/  *.png
```

Pa zmoŭčańni vyjavy čytajucca ŭ pamiery 28×28 pikseliaŭ i transfarmujucca va ŭvachodny adnamierny vektar pamieram u 784 elementy. Abodva pamiery naładžvajucca praz `--dataset-img-width` i `--dataset-img-height` — hałoŭnaje, kab `šyrynia × vyšynia` supadała z uvachodnym słojem sietki.

## Dakumentacyja (pa-angieĺsku)

- [Zborka](docs/building.md) — opcyi CMake, mety zborki, zaliežnaści, testy
- [Klasifikatar](docs/classifier.md) — apisańnie navučańnia i inferensu ŭtylitaj kamandnaha radku
- [GAN](docs/gan.md) — navučańnie GAN i generacyja vyjaŭ utylitaj kamandnaha radku
- [Viektarnyja pradstaŭleńni słovaŭ](docs/words.md) — kanvejer SGNS, struktura moduliaŭ, apisańnie CLI
- [Hiperparametry](docs/hyperparameters.md) — rasklad uzroŭniu navučańnia (learning rate), dapamožnik pa architektury, parady dlia stabilizacyi GAN
- [Grafičny interfejs](docs/gui.md) — režymy, patoki, supolny terminał

</details>
