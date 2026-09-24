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

## What the wire costs

Measured against a Release build, one model 784-128-10, `--threads 2`, one
keep-alive client, median of three thousand requests:

| Representation | Body | Median | Throughput |
|---|---|---|---|
| `application/octet-stream` | 6272 B | 151 us | 6600 req/s |
| `application/json` | 15043 B | 561 us | 1800 req/s |

The binary path is 3.7x quicker for the same prediction, and the gap is the
decode: nlohmann has to build a document tree, the raw path is a loop of
`memcpy`. That is why `octet-stream` exists at all, and why JSON stays — it is
the only representation a human can send by hand, and nothing in this step ships
a tool that builds a frame.

**`TCP_NODELAY` is the server's job and only the server's.** httplib defaults
`CPPHTTPLIB_TCP_NODELAY` to false. Measured over all four combinations of the
flag on a keep-alive binary round trip:

| Server | Client | Median |
|---|---|---|
| on | on | 91.9 us |
| on | off | 84.1 us |
| off | on | 41162 us |
| off | off | 41324 us |

With the server setting it, what the client does is noise. With the server not
setting it, nothing the client does helps: it is Nagle against a delayed ACK on
the response, and forty milliseconds is the delayed-ACK timer. So a client needs
no special socket handling, and `ServerSettings::tcpNoDelay` is asserted by a
test rather than trusted, because losing it leaves a service that answers
correctly and ninety times slower.

## Two listeners, and why

`--port` carries `/healthz`, `/v1/models` and the predict routes. `--admin-port`,
on loopback by default, carries `/healthz`, `/readyz` and `POST /admin/reload`.

Authentication and TLS are out of scope for this step, which makes the bind
address the only access control there is. `/readyz` publishes absolute server
paths and the text of manifests — which is operator input, echoed through an
exception message — and `/admin/reload` triggers an unauthenticated re-read of the
whole tree. Neither belongs on a public socket, and a test asserts that neither
answers there.

## Readiness

`/healthz` is liveness and never reads the registry: a probe that can fail for a
reason a restart will not fix is a probe that gets the process killed. It answers
one question, whether the listener accepted and routed.

`/readyz` is readiness, and its predicate is:

```
default:         generation > 0 && !models.empty()
--strict-ready:  generation > 0 && !models.empty() && failures.empty()
```

`generation > 0` is the only thing that separates *never published* from
*published an empty composition*: the registry's constructor installs an empty
snapshot at generation 0 and `install` stamps `++published`.

`failures.empty()` is deliberately **not** the default gate. Step 2 decided that
the unit of failure is a directory and the unit of value is a model, and that they
are independent; gating readiness on the failure list re-couples exactly that, so
one unparsable directory would pull a healthy nineteen-model service out of
rotation. `--strict-ready` is for operators who want them coupled, and the same
flag also makes `serve` exit `kIncomplete` before binding rather than start
degraded — one flag, both meanings.

The body has the same shape at 200 and at 503, so a probe never branches on the
status to read it, and it is bounded: a reason is elided at 512 bytes and at most
32 failures are printed, the rest counted in `detailTruncated`. Without that, a
manifest whose `input` is a 900 KB array would put 900 KB into a readiness probe.

**A broken `--default` is reported with `directory` set to the root**, not to a
model directory, because the mistake is in the configuration rather than in any
artifact. It does carry `name` and `version`, which a parse failure cannot. A
reader that renders every failure as "this directory did not load" will misreport
a typo as a corrupt tree.

## Reload

Both triggers, one path: `POST /admin/reload` and `SIGHUP` call the same function.

It is synchronous. `rebuild()` re-reads everything — measured in
`core/serving/README.md` at 1.3 ms for a 795 KB model and about 0.03 s for twenty
MNIST-sized ones — the admin listener has its own small pool so a reload never
waits behind public traffic, and its write timeout is raised to 60 s so even a
pathological tree answers rather than dropping the connection.

One rebuild at a time, refused with 409 rather than queued: a second rebuild would
re-read a tree the first one is reading right now.

**That guarantee needs the admin listener to have two threads, and the reason is
not obvious.** A pool task is a whole connection, not a request, so a
single-threaded admin listener cannot hold two reload requests at once: the second
waits in the kernel backlog until the first has finished and released the gate,
then gets a thread, finds the gate free, and pays for a full second re-read.
Measured on twelve 5 MB models, one rebuild costing 141 ms, two simultaneous
`curl`s:

| Admin threads | Wall clock | First | Second |
|---|---|---|---|
| 1 | 280 ms | 200, generation 2 | 200, generation 3 |
| 2 | 171 ms | 200, generation 3 | **409** |

At one thread the 409 in this document did not exist over HTTP, and the
documentation was describing something the code could not do.

The other way in is genuinely concurrent whatever the pool looks like: the signal
thread is not part of it. Ten rounds of a `SIGHUP` racing an HTTP reload published
exactly ten generations for twenty requests — every HTTP request answered 409 while
the signal thread held the gate.

A failed rebuild is non-destructive, because publication is the last statement,
and the **unchanged `generation` in the 500 body is the proof**. The handler
catches `std::exception`, not only `Serving::Error`: `Build` catches only the
latter, so a `bad_alloc` or an uncovered `filesystem_error` escapes it.

`SIGHUP` goes through a self-pipe. The handler writes one byte and touches nothing
else — no mutex, no allocation, no stream, all of which are undefined there — and
a dedicated thread does the rebuild. `SIGINT` and `SIGTERM` write a different byte
through the same pipe, which is how the process stops cleanly.

**Rapid `SIGHUP`s coalesce.** Standard signals are not queued, so two arriving
before the handler runs produce one rebuild. That is the desirable outcome — one
re-read instead of two identical ones — but it means `SIGHUP` is a request to
reconverge, not a counter.

## Batching, and where the ceiling comes from

`forward` is batched by rows, and one request carrying several rows amortises both
the inference and the HTTP work around it. Measured end to end, Release, binary
frame, 784-128-10, server pinned to two CPUs and the client to two others, median
of three runs:

| Rows | Median | Per row | Against one row | Body |
|---|---|---|---|---|
| 1 | 172 us | 172 us | 1.00x | 6 KB |
| 8 | 287 us | 36 us | 4.8x | 49 KB |
| 16 | 515 us | 32 us | 5.4x | 98 KB |
| 32 | 875 us | 27 us | **6.9x** | 196 KB |
| 64 | 2620 us | 41 us | 4.2x | 392 KB |

`--max-batch-rows` defaults to **32**, and it is a hard refusal — 413
`batch_too_large` — not a policy switch. Thirty-two is where both arguments land:
it is the best per-row cost measured, and it is the largest batch that still fits
the tail. A request holds one connection slot for its whole length, the measured
p99 under saturation is about 1200 us, and 32 rows come in at 875 us where 64 rows
take 2620 us. Memory is nowhere near binding: 32 rows of the heaviest measured
topology carry about 400 KB of intermediates.

**Where the rise past 32 rows is, and where it is not.** It is not the inference:
`Neural::Predict` per row improves monotonically all the way out — 48 us at one
row, 8.6 us at 64 — and it is not the handler either, whose decode, fill, predict
and encode together hold flat at about 10.7 us per row from 8 rows to 64. What is
left is the transport of the body itself, which doubles from 196 KB to 392 KB
across that step. The cost is measured; its mechanism is not, and this file will
not guess at one.

That distinction was worth the measurement: the first explanation written here
blamed cache pressure on the activations, which the split above refutes.

**The fill and encode loops walk columns, not rows, and that is not cosmetic.**
`Matrix` wraps a column-major Eigen matrix, so a row-major walk strides by the row
count and gets worse the taller the batch. Measured on a 784-column fill: about
2.2 us per row either way at 8 rows, but 7.1 against 2.2 at 128. Written the
row-major way the handler's own overhead grew from 3.1 to 6.9 us per row between 8
and 64 rows; written this way it stays near 3.

**These absolute numbers drift.** The single-row path measured 41.7 us, 48.2 us and
59.9 us for the same topology across three sittings on the same laptop, a CPU that
throttles differently from hour to hour. Ratios held every time, which is why the
derivations here rest on ratios and on a p99 measured in the same sitting.

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
