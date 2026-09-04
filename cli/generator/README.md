# `cli/generator` — the `MatrixGui_gan` subcommands

The command-line front end over [`core/generator`](../../core/generator/) and
[`core/nn`](../../core/nn/). Two subcommands: `train` and `generate`.

The user-facing flag reference lives in [`docs/gan.md`](../../docs/gan.md) and is
not repeated here; this file is about how the pieces fit. The layout follows
[`cli/words`](../words/README.md), with one deliberate difference — see
[Errors and exit codes](#errors-and-exit-codes).

| File | Contains |
|---|---|
| `options.h` | One options struct per subcommand |
| `commands.h` | One `int Name(std::ostream&, std::ostream&, const NameOptions&)` per subcommand, and the exit-code constants |
| `commands.cpp` | The bodies, plus the local helpers they share |
| `main.cpp` | The CLI11 app: flag definitions and dispatch |

## Its own library

`commands.cpp` builds into `matrixgui_cli_generator`; `main.cpp` stays out of it,
so the subcommand bodies can be linked into a test binary without dragging in a
second `main()`:

```cmake
add_library(matrixgui_cli_generator ${MATRIXGUI_LIB_TYPE} commands.cpp)
add_executable(MatrixGui_gan main.cpp)
```

Keeping this separate from `matrixgui_generator` is what stops CLI11 from
reaching the Qt binary, which links `matrixgui_generator` but not this.

## Dispatch

Each subcommand binds its flags to its own options struct and attaches its body
with CLI11's `->callback()`:

```cpp
trainCmd->callback([&] {
    exitCode = GeneratorCli::Train(std::cout, std::cerr, train);
});
```

`app.require_subcommand(1)` sets both the minimum and the maximum to one, so
exactly one callback runs and assigns `exitCode`.

`CLI11_PARSE` is deliberately not used. It expands to a `try`/`catch` that covers
only `CLI::ParseError`, and the bodies now run *inside* `app.parse()`. The
explicit form calls the same `app.exit()`, so a parse failure still exits 106 and
`--help` still exits 0.

There is deliberately **no** catch-all around `app.parse()`. `Train` does not
catch, so a failure loading the dataset or writing a network aborts the process,
and that is the existing behaviour. A `catch (const std::exception&)` here would
silently turn those aborts into exit code 4.

## The shape of a command body

Both bodies take `(out, err, options)` and return the process exit code. The
options struct is a **verbatim record of the flags** — every derivation lives in
the body.

The layer topologies are the worked example. `TrainOptions` stores only the
hidden-layer lists the flags bind to; `BuildGeneratorLayers` and
`BuildDiscriminatorLayers` in `commands.cpp` assemble the full stacks, because the
generator's input width is `latentDim + classesCount` and `classesCount` is only
known once the frozen classifier has been loaded.

## Errors and exit codes

Unlike [`cli/words`](../words/README.md), whose commands return `void` and throw,
these return an `int`. The codes are semantic and documented, so they are
returned from the place that makes the decision. The named constants are in
`commands.h`; the table for users is in
[`docs/gan.md`](../../docs/gan.md#exit-codes).

Two things that table does not say:

- `Generate` catches `std::exception` itself and maps it to `kGenerateFailed` (4).
- `Train` does **not** catch. An exception escaping it terminates the process.

One quirk that follows from the registration in this file: `--generator` has both
a default (`generator.wgt`) *and* a `->check(CLI::ExistingFile)`. CLI11 does not
apply a check to a default, so running `generate` in a directory without a
`generator.wgt` reaches the body and exits `2`, not `106`.

## Shared with the classifier

The LRU-cached PNG reader is [`cli/lib/png_reader.h`](../lib/png_reader.h), used
by `MatrixGui_headless` too. Its cache lives behind a `shared_ptr` because
`cache::LRUCache` is non-copyable and `std::function` requires a copyable target.

## Adding a subcommand

1. Add an options struct to `options.h`.
2. Declare and define `int Name(std::ostream&, std::ostream&, const NameOptions&)`.
3. Register the flags and a `->callback()` that assigns `exitCode` in `main.cpp`.
4. Add any new exit code to `commands.h` **and** to the table in
   [`docs/gan.md`](../../docs/gan.md#exit-codes).
