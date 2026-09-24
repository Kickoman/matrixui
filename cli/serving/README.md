# `cli/serving` — the `MatrixGui_models` subcommands

The front end over [`core/serving`](../../core/serving/). Two subcommands:
`list`, which walks a directory of model artifacts and prints what loaded and why
the rest did not, and `serve`, which holds the same composition in a registry and
answers over HTTP.

`core/serving/README.md` is the reference for the manifest format, the loader's
checks, the registry and the snapshot; nothing here repeats it. This file is
about how the pieces fit, about the exit codes, which are this tool's own, and
about what the `serve` limits cost.

| File | Contains |
|---|---|
| `options.h` | `struct ListOptions`, `struct ServeOptions` — plain fields, no accessors |
| `commands.h` | One `int Name(std::ostream&, std::ostream&, const NameOptions&)` per subcommand, and the exit-code constants |
| `commands.cpp` | The bodies, the table formatting and the failure report they share |
| `handlers.h/.cpp` | The request layer as pure functions: no httplib, no sockets, `(snapshot, request) -> reply` |
| `server.h/.cpp` | The only translation unit that includes httplib: both listeners, routes, settings |
| `main.cpp` | The CLI11 app: flag definitions, `--default` parsing, dispatch |

`handlers` and `server` are split so that every status code is reachable from a
doctest without a socket, and so that the fourteen seconds httplib costs to
compile are paid by one file.

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

## What the `serve` limits cost

`--max-body-bytes` defaults to 8 MiB against httplib's own
`CPPHTTPLIB_PAYLOAD_MAX_LENGTH` of 100 MB, and the reason is not tidiness. The
cap bounds what a hostile `Content-Length` makes the process buffer *before* a
handler runs, and for the JSON representation the buffer is the cheap part.

Measured, Release flags, one core, a body of empty rows at the cap:

| Body | Peak resident | CPU | Answer |
|---|---|---|---|
| 7.98 MiB, 2 790 000 rows | 236 MiB | ~0.5 s | 413 `batch_too_large` |
| 5.72 MiB, 2 000 000 rows | 164 MiB | ~0.3 s | 413 `batch_too_large` |

That is nlohmann's document tree, and it is **not avoidable by checking things
sooner**: the row count only exists once the body has been parsed. At
`--threads 32` the worst case is about thirty times the first row, so a public
port and a large `--max-body-bytes` are a memory budget, not a formality. 8 MiB
is chosen so that the row ceiling binds first for any plausible model — 64 rows
of 784 values is 401 KB — and so that thirty-two concurrent worst cases stay in
the same order as the model set the registry already holds.

**The two representations do not cost the same for the same byte cap.** A binary
body allocates in proportion to itself, because the row count is
`body.size() / (input.size * 8)`; a JSON body of the same length can declare
millions of rows. What the cap can do is refuse the count before anything is
sized from it, which is where the 413 above comes from. What it cannot do is make
the parse cheaper.

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
