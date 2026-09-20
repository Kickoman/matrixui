# MatrixGui

Neural network tooling built from scratch, C++20. See [README.md](README.md) for
what it does and [docs/](docs/) for the narrative documentation.

## Comments

**Code carries as few comments as possible; ideally none.** Explanations belong
in the README of the module they are about, not scattered through the source.

Acceptable in code:
- a short technical note a reader cannot get from the code itself (a byte
  offset, a units convention, a platform quirk);
- a genuinely exceptional decision, where the obvious reading of the code is
  wrong and the reason is not written anywhere else.

Not acceptable: restating what the code does, design rationale, "why not the
other approach", anything that is really a paragraph of documentation. That goes
to the module README.

Every folder that is a module of its own gets a `README.md` — see
[core/words/README.md](core/words/README.md) and its per-folder children for the
shape: what each unit exposes, what it guarantees, and what will bite you.

## Style

- Free functions `PascalCase`, methods `camelCase`.
- Declarations in `.h`, definitions in `.cpp`.
- One namespace per module (`Neural`, `Words`, `Serving`, `Io`, `Hash`, `Text`).
- Errors are exceptions, with a per-module hierarchy rooted at a module `Error`.
- Builds must stay clean under `-Wall -Wextra`.

## Tests

`tests/` is doctest plus the golden CLI snapshots in `tests/golden/`.

```bash
cmake -B build -DBUILD_TESTS=ON && cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
tests/golden/compare.sh
```

Golden snapshots are recorded on the machine that runs them and are not part of
CI. Do not re-record an existing expectation to make a change pass; adding new
cases is fine.

## Git

Commits and pushes are the user's, never the assistant's.
