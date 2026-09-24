# `cli/serving` — the `MatrixGui_models` subcommands

The operator-facing front end over [`core/serving`](../../core/serving/). One
subcommand today: `list`, which walks a directory of model artifacts and prints
what loaded and why the rest did not.

`core/serving/README.md` is the reference for the manifest format, the loader's
checks, the registry and the snapshot; nothing here repeats it. This file is
about how the two pieces fit and about the exit codes, which are this tool's own.

| File | Contains |
|---|---|
| `options.h` | `struct ListOptions` — plain fields, no accessors |
| `commands.h` | `int List(std::ostream&, std::ostream&, const ListOptions&)` and the exit-code constants |
| `commands.cpp` | The body, plus the table formatting it uses |
| `main.cpp` | The CLI11 app: flag definitions, `--default` parsing, dispatch |

## Its own library

`commands.cpp` builds into `matrixgui_cli_serving` and `main.cpp` stays out of
it, so the body can be linked into `MatrixGui_tests` without a second `main()` —
the same split [`cli/classifier`](../classifier/README.md) uses and for the same
reason. The library is built unconditionally; only the executable is behind
`BUILD_CLI`, because `-DBUILD_CLI=OFF -DBUILD_TESTS=ON` must still link the body.

`List` takes both streams as arguments rather than writing to `std::cout`
directly, which is what lets `tests/cli/serving_commands_test.cpp` assert the
whole surface without starting a process.

## The stdout / stderr split

The inventory goes to stdout, every reason a directory did not load goes to
stderr. That is deliberate and it is what makes the tool usable in a pipeline: a
caller can take stdout as the inventory and still see the problems. The two are
never interleaved into one stream.

## Errors and exit codes

| Code | Constant | Means |
|---|---|---|
| 0 | `kSuccess` | Everything under the root loaded |
| 1 | `kIncomplete` | The composition is incomplete — something is in `failures` |
| 2 | `kUnusableRoot` | The root itself could not be walked |
| 3 | `kBadOption` | A malformed `--default`, which is parsed by hand |
| 105 / 106 / 109 | *(CLI11's)* | Flag parsing, owned by `app.exit()` |

`1` is the one a deployment check gates on: it means the tool worked and the tree
did not. It is distinct from `2` because an unwalkable root is not a partial
result — `Build` throws rather than returning half a composition.

The CLI11 codes are not ours but they are **pinned** by
`tests/golden/serving/expected/*.code`, so a new exit code must not collide with
them.

## What will bite you

**`--default` is parsed by hand, in `main.cpp`'s callback.** CLI11 has no
`name=value` validator here, so the split on `'='` and the three ways it can be
malformed (no `=`, empty name, empty version) live in the callback and return
`kBadOption`. That parsing has no unit test of its own — only the golden
snapshot covers it.

**A default naming a version that did not load is a failure, not a refusal.** It
lands in `failures` with `FailureKind::Manifest` and `directory` set to the
*root*, not to a model directory, because the typo is in the configuration rather
than in any artifact. A reader that renders every failure as "this directory did
not load" will misreport it.

**The paths in refusals are absolute.** They come back from `weakly_canonical`,
so they carry the checkout prefix and are machine-specific; the golden capture
script normalises them away, and anything else that publishes them has to decide
whether it should.
