# The desktop application

`MatrixGui` is the Qt front end. It does the same work as the four CLI binaries,
but interactively: you pick a mode, fill in a form, press a button, and watch the
training log scroll past in a terminal pane.

The window is a set of tabs. A tab always starts as the **New mode** picker — a
3×2 grid of large buttons — and turns into whichever mode you choose. You can
open several tabs and run different things in each.

```bash
./build/MatrixGui              # follows the system light/dark setting
./build/MatrixGui --theme dark # force one
```

## Modes

| Button | Mode | CLI equivalent |
|--------|------|----------------|
| Digits classifier | Train a classifier, watch loss and accuracy | [`MatrixGui_headless train`](classifier.md) |
| Digits recognition playground | Draw a digit with the mouse, or load a PNG, and classify it | [`MatrixGui_headless predict`](classifier.md) |
| GAN generative network training | Train a conditional GAN and generate samples | [`MatrixGui_gan`](gan.md) |
| Word embeddings (SGNS) | Build a vocabulary and corpus, train, then explore and evaluate | [`MatrixGui_words`](words.md) |
| Functions symbolic regression | Place data points on a plot and evolve an expression fitting them | [`MatrixGui_functions`](../cli/functions/README.md) |

Every mode is a pair: a **controller** that owns the work and a **widget** that
shows it. `CreateMode()` in `gui/lib/mode_factory.cpp` is the single place they
are wired together, including pointing the controller's log output at the
widget's terminal pane.

Because the controllers drive the same core classes the CLI does, and the output
goes through the same `report/` printers, what you see in the GUI terminal is
byte-for-byte what the CLI would have printed.

<details>
<summary>Threading contract</summary>

This is the invariant the GUI rests on, and nothing in the type system enforces
it — so it is worth knowing before you touch `gui/`.

**Every controller member is written on the GUI thread only.** Long operations
run on a worker thread, and there is at most one such thread per controller at a
time. A worker never touches controller state directly. Instead it:

1. captures immutable snapshots up front — `shared_ptr`s to const data, copies of
   paths — before it starts;
2. publishes its result back with `QMetaObject::invokeMethod`, which lands on the
   GUI thread.

The busy flag is set on the GUI thread *before* the worker starts, not inside the
worker, so a fast double-click cannot slip a second run through the gap.

The core's own progress callbacks make the same demand from the other direction.
`Words::Trainer` calls its progress callback **on the thread that called
`train()`**, not on the UI thread, so a controller must marshal it. This is the
easiest thing in the GUI to get silently wrong; it will appear to work and then
crash under load.

**Read widget values at click time, never on `valueChanged`.** Config structs are
assembled when the button is pressed. Reacting to `valueChanged` instead creates a
`setValue` → `valueChanged` → `infoUpdated` → `setValue` re-entrancy loop, because
refreshing the info panel writes back into the same widgets. Widgets are seeded
once, at construction.

</details>

<details>
<summary>The shared terminal</summary>

`gui_common/advanced_terminal.h` is what lets framework-free core code print into
a Qt widget. `AdvancedTerminal` wraps a `QPlainTextEdit`;
`ThreadSafeTerminalOStream` is a real `std::ostream` on top of it, so anything
taking a `std::ostream&` — every printer in `core/words/report/`, every trainer's
verbose output — can write to it unchanged.

Two behaviours are worth knowing:

- **A completed line is shown as soon as it is written.** This is not the obvious
  default. `std::endl` reaches a `streambuf` as a `'\n'` plus a `sync()` call,
  never as a manipulator, so a stream facade that only flushes on the `std::endl`
  *manipulator* will leave output invisible whenever the writer ends lines with a
  plain `'\n'` — which the report printers do. The stream therefore flushes on
  every completed line.
- **A carriage return rewrites the current line** rather than appending, so
  progress counters that use `\r` animate in place instead of filling the pane.

In the Words mode, all three sub-tabs share a single terminal that sits below
them and stays visible whichever tab is active.

</details>

<details>
<summary>Theme detection</summary>

`AppTheme::IsSystemDarkMode()` (`gui/lib/theme.cpp`) decides the default theme.
On Linux it asks `gsettings` for the desktop's `color-scheme` first, and falls
back to reading `gtk-theme` and looking for "dark" in the name — the fallback
exists because Ubuntu 24.04 does not report `color-scheme` the way newer
desktops do. When `gsettings` is unavailable it returns nothing and the light
theme is used.

`--theme dark` / `--theme light` overrides all of that. The dark palette itself
comes from the vendored `contrib/qt-dark-theme` subproject. Charts are themed
separately through `AppTheme::ApplyTheme(QChart*)`, since Qt Charts does not pick
up the application stylesheet.

</details>

<details>
<summary>Settings persistence</summary>

`ModeSettings` (`gui/lib/mode_settings.h`) is a thin wrapper over `QSettings`
that prefixes every key with the mode's name, so two modes can both store
`"datasetPath"` without colliding. The organisation and application names set in
`gui/main.cpp` decide where the file lands (`~/.config/Kastus/Networks.conf` on
Linux).

It stores form contents — paths, spin-box values — not models. Weights and
embeddings are always explicit files you save and load.

</details>

<details>
<summary>Adding a new mode</summary>

Four edits, all mechanical:

1. Add an entry to `enum class ModeType` in `gui/lib/mode_factory.h`.
2. Write a controller deriving from `ModeController` and a widget deriving from
   `ModeWidget`. If the mode does long work, override `requestStop()` and
   `waitUntilFinished()` so closing a tab does not abandon a running thread.
3. Add a `CreateYourMode()` helper and a `switch` case in
   `gui/lib/mode_factory.cpp`. If the mode logs, call
   `controller->setLogger(view->getTerminalStream())` there.
4. Add a button in `gui/lib/new_mode_widget.cpp` that emits
   `modeRequested(ModeType::YourMode)`.

The picker's 3×2 grid is deliberate: three fixed 300 px buttons need about
940 px, which fits, while a fourth in the same row would overflow any window
narrower than about 1260 px. The sixth cell is free; a seventh mode means
rethinking that layout rather than appending to it.

Add the new sources to the `add_executable(MatrixGui ...)` list in
`gui/CMakeLists.txt`, and link the core library the mode needs. Keep the actual work in
`core/` — taking a `std::ostream&` rather than printing to `std::cout` — so it
stays usable from a CLI too.

</details>

## See also

- [Classifier](classifier.md), [GAN](gan.md), [Word embeddings](words.md), [Functions](../cli/functions/README.md) — what each mode actually does
- [Building](building.md) — Qt requirements and the `BUILD_GUI` option
