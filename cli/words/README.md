# `cli/words` — the `MatrixGui_words` subcommands

The command-line front end over [`core/words`](../../core/words/README.md). Four files,
and deliberately flat: one struct per subcommand's options, one function per
subcommand's body.

| File | Contains |
|---|---|
| `options.h` | One options struct per subcommand |
| `commands.h` | One `void Name(std::ostream&, const NameOptions&)` per subcommand |
| `commands.cpp` | The bodies, plus two shared local printers |

The flag definitions themselves are in `main.cpp`, which builds the CLI11 app.

The user-facing flag reference is in [`docs/words.md`](../../docs/words.md); this
file is about how the pieces fit.

## Its own library

`commands.cpp` builds into `matrixgui_cli_words`; `main.cpp` stays out of it, so
the subcommand bodies can be linked into a test binary without dragging in a
second `main()`:

```cmake
add_library(matrixgui_cli_words ${MATRIXGUI_LIB_TYPE} commands.cpp)
target_link_libraries(matrixgui_cli_words PUBLIC matrixgui_base matrixgui_words)

add_executable(MatrixGui_words main.cpp)
target_link_libraries(MatrixGui_words PRIVATE matrixgui_cli_words matrixgui_contrib)
```

Keeping this separate from `matrixgui_words` is what stops CLI11 from reaching
the Qt binary, which links `matrixgui_words` but not this. Adding a file here
means editing `cli/words/CMakeLists.txt`.

## Dispatch

Each subcommand binds its flags to its own options struct and attaches its body
with CLI11's `->callback()`:

```cpp
inspectCmd->callback([&] { WordsCli::Inspect(std::cout, inspect); });
```

So CLI11 calls the right handler directly. There is no switch, no map from name
to function, and nothing to keep in step with the registrations — a subcommand
that is registered is wired, or it does not compile.

`app.require_subcommand(1)` makes a bare invocation an error rather than a
silent no-op.

## Shape of a command body

Every body is a few lines: load what it needs, call into `core/words`, hand the
resulting struct to a printer from `core/words/report/`.

```cpp
void Inspect(std::ostream& out, const InspectOptions& options) {
    Words::PrintCorpusStatistics(out, Words::InspectDump(options.input, options.topN));
}
```

**Every body takes the output stream as a parameter.** `main` passes
`std::cout`; the tests pass a `std::ostringstream`. Nothing here writes to a
stream it found on its own.

Two local helpers in `commands.cpp` — `PrintVocabularyInfo` and
`PrintCorpusInfo` — print the short summaries the build/load commands share.
They live here rather than in `report/` because they describe a *CLI step*, not
a report struct the GUI would ever render.

## Errors and exit codes

Nothing here catches. `main` catches `Words::Error` once and exits non-zero, so
a body may let an `IoError` from a missing file propagate.

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | a failed check or reported error |
| `2` | an internal error |
| `106` | the command line did not parse — CLI11's own code |

CLI11 rejects a missing required option, an unknown flag, an absent subcommand,
and — via `->check(CLI::ExistingFile)` — an input file that does not exist,
before any body runs.

## Adding a subcommand

1. Add an options struct to `options.h`.
2. Declare and define `void Name(std::ostream&, const NameOptions&)`.
3. Register the flags and a `->callback()` in `main.cpp`.
4. Extend `tests/golden/capture.sh` so the new subcommand is snapshotted, and
   re-run it to record the expected output.

Keep computation out of the body. If a command needs logic worth testing, it
belongs in `core/words` where the unit tests can reach it — this layer is not in
the test binary.
