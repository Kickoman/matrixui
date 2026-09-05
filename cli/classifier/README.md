# `cli/classifier` — the `MatrixGui_headless` subcommands

The command-line front end over [`core/classifier`](../../core/classifier/) and
[`core/nn`](../../core/nn/). Two subcommands: `train` and `predict`.

The user-facing flag reference lives in
[`docs/classifier.md`](../../docs/classifier.md) and is not repeated here; this
file is about how the pieces fit. The layout follows
[`cli/words`](../words/README.md), with one deliberate difference — see
[Errors and exit codes](#errors-and-exit-codes).

| File | Contains |
|---|---|
| `options.h` | One options struct per subcommand, plus `DefaultNetworkConfiguration()` |
| `commands.h` | One `int Name(std::ostream&, std::ostream&, const NameOptions&)` per subcommand, and the exit-code constants |
| `commands.cpp` | The bodies, plus the local helpers they share |
| `main.cpp` | The CLI11 app: flag definitions, the config pre-scan, dispatch |

## Its own library

`commands.cpp` builds into `matrixgui_cli_classifier`; `main.cpp` stays out of
it, so the subcommand bodies can be linked into a test binary without dragging
in a second `main()`:

```cmake
add_library(matrixgui_cli_classifier ${MATRIXGUI_LIB_TYPE} commands.cpp)
add_executable(MatrixGui_headless main.cpp)
```

Keeping this separate from `matrixgui_classifier` is what stops CLI11 from
reaching the Qt binary, which links `matrixgui_classifier` but not this.

## Dispatch

Each subcommand binds its flags to its own options struct and attaches its body
with CLI11's `->callback()`:

```cpp
trainCmd->callback([&] {
    exitCode = ClassifierCli::Train(std::cout, std::cerr, train);
});
```

`app.require_subcommand(1)` sets both the minimum and the maximum to one, so
exactly one callback runs and assigns `exitCode`.

`CLI11_PARSE` is deliberately not used. It expands to a `try`/`catch` that covers
only `CLI::ParseError`, and the bodies now run *inside* `app.parse()`. The
explicit form calls the same `app.exit()`, so a parse failure still exits with
CLI11's own code (106 missing required option or subcommand, 105 failed file
check, 109 unknown flag) and
`--help` still exits 0.

There is deliberately **no** catch-all around `app.parse()`. `Train` does not
catch, so an `Io::Error` — an unwritable `--working-directory`, say — aborts the
process, and that is the existing behaviour. A `catch (const std::exception&)`
here would silently turn those aborts into exit code 4.

## The config pre-scan

This is the one piece of ordering a reader can break without a word from the
compiler.

```cpp
train.learningConfigPath = CliLib::FindOptionValue(argc, argv, "--learning-config");
train.networkConfigPath  = CliLib::FindOptionValue(argc, argv, "--network-config");
if (!CliLib::LoadJsonConfig(std::cerr, train.learningConfigPath, train.learning, "learning")) { ... }
```

The config files are read straight out of `argv`, **before** the per-field
options are registered. Two things depend on that:

1. **Override semantics.** The file writes into `train.learning` /
   `train.network`, the same objects the flags later bind to. CLI11 leaves a
   bound variable untouched when its flag is absent, so the file supplies the new
   defaults and an individual flag still overrides just that field.
2. **Honest `--help`.** The `capture_default_str()` calls below snapshot each
   bound variable at registration time, so the help text shows the values that
   will actually be used.

A consequence worth knowing: a `--learning-config` path that does not exist exits
`4` here, before CLI11's `->check(CLI::ExistingFile)` could turn it into `105`.

## The shape of a command body

Both bodies take `(out, err, options)` and return the process exit code. The
options struct is a **verbatim record of the flags** — every derivation,
fallback and validation lives in the body. The dataset fallback is the worked
example: `options` keeps three independent strings, and `Train` resolves them:

```cpp
const std::string trainingPath = !options.trainDatasetPath.empty() ? options.trainDatasetPath : options.datasetPath;
const std::string testingPath  = !options.testDatasetPath.empty()  ? options.testDatasetPath  : options.datasetPath;
if (trainingPath.empty() || testingPath.empty()) {
    err << "Specify data directories: ...";
    return kNoDatasetResolved;
}
```

An accessor on the struct would put the fallback in `options.h` and the exit code
that depends on it in `commands.cpp`, splitting one decision across two files.

## Errors and exit codes

Unlike [`cli/words`](../words/README.md), whose commands return `void` and throw,
these return an `int`. The codes are semantic and documented, so they are
returned from the place that makes the decision. The named constants are in
`commands.h`; the table for users is in
[`docs/classifier.md`](../../docs/classifier.md#exit-codes).

Two things that table does not say:

- `Predict` catches `std::exception` itself and maps it to `kBadConfig` (4).
- `Train` does **not** catch. An exception escaping it terminates the process.

## Adding a subcommand

1. Add an options struct to `options.h`.
2. Declare and define `int Name(std::ostream&, std::ostream&, const NameOptions&)`.
3. Register the flags and a `->callback()` that assigns `exitCode` in `main.cpp`.
4. Add any new exit code to `commands.h` **and** to the table in
   [`docs/classifier.md`](../../docs/classifier.md#exit-codes).

## Do not

- **Do not unify `activationMap` with the enum in `core/nn/layers.h`.** That
  enum's `NLOHMANN_JSON_SERIALIZE_ENUM` accepts `leakyrelu`, so a
  `--network-config` may name it; the CLI map deliberately does not, so
  `--hidden-activation leakyrelu` stays rejected.
- **Do not move the config pre-scan** below the option registrations.
- **Do not give `imageWidth`/`imageHeight` back to a single shared variable.**
  They used to be one pair serving both subcommands; each options struct now
  owns its own.
