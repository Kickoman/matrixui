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
inspectCmd->callback([&] {
    exitCode = WordsCli::Inspect(std::cout, std::cerr, inspect);
});
```

So CLI11 calls the right handler directly. There is no switch, no map from name
to function, and nothing to keep in step with the registrations — a subcommand
that is registered is wired, or it does not compile.

`app.require_subcommand(1)` sets both the minimum and the maximum to one, so a
bare invocation is an error and exactly one callback assigns `exitCode`.

## Shape of a command body

Every body is a few lines: load what it needs, call into `core/words`, hand the
resulting struct to a printer from `core/words/report/`.

```cpp
void RunInspect(std::ostream& out, const InspectOptions& options) {
    Words::PrintCorpusStatistics(out, Words::InspectDump(options.input, options.topN));
}

int Inspect(std::ostream& out, std::ostream& err, const InspectOptions& options) {
    return Guarded(err, [&] { RunInspect(out, options); });
}
```

**Every body takes its streams as parameters.** `main` passes `std::cout` and
`std::cerr`; the tests pass `std::ostringstream`s. Nothing here writes to a
stream it found on its own.

Two local helpers in `commands.cpp` — `PrintVocabularyInfo` and
`PrintCorpusInfo` — print the short summaries the build/load commands share.
They live here rather than in `report/` because they describe a *CLI step*, not
a report struct the GUI would ever render.

## Errors and exit codes

The bodies do not catch; they signal failure by throwing `Words::Error` or
`Io::Error`. The `Guarded` wrapper in `commands.cpp` catches once for all
eleven commands, prints the `error: ` / `internal error: ` line to `err`, and
returns the code. The constants live in `commands.h`; `main` keeps only the
`CLI::ParseError` catch, exactly like the classifier and generator CLIs.

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | a failed check or reported error |
| `2` | an internal error |
| `105`/`106`/`109` | CLI11's own parse-failure codes: failed file check / missing required option or subcommand / unknown flag |

CLI11 rejects a missing required option, an unknown flag, an absent subcommand,
and — via `->check(CLI::ExistingFile)` — an input file that does not exist,
before any body runs.

## Adding a subcommand

1. Add an options struct to `options.h`.
2. Define `RunName(out, options)` and wrap it:
   `int Name(std::ostream&, std::ostream&, const NameOptions&)` via `Guarded`.
3. Register the flags and a `->callback()` that assigns `exitCode` in `main.cpp`.
4. Extend `tests/golden/words/capture.sh` so the new subcommand is snapshotted, and
   re-run it to record the expected output.

Keep computation out of the body. If a command needs logic worth testing, it
belongs in `core/words` where the unit tests can reach it — this layer is not in
the test binary.
