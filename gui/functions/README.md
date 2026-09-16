# `gui/functions` — the Functions mode

The Qt front end over [`core/functions`](../../core/functions/): place data
points, press Start, and watch the population evolve expressions that fit them.
The CLI equivalent is [`MatrixGui_functions run`](../../cli/functions/README.md),
and the two read and write the same config JSON and the same points CSV.

| File | Contains |
|---|---|
| `functions_controller.{h,cpp}` | Owns the session and the worker thread; publishes snapshots |
| `functions_mode_widget.{h,cpp}` | The whole mode: layout, wiring, the world table |
| `functions_config_widget.{h,cpp}` | Spinbox form over `Genetizer::FunctionsConfig` |
| `functions_points_model.{h,cpp}` | The data points, the single source of truth behind the table and the plot |
| `function_plot_widget.{h,cpp}` | The interactive plot: axes, pan, zoom, point editing, curve drawing |

## The session, and what Start / Resume / Restart mean

A GUI run has no epoch limit: it evolves until Stop. What survives a Stop is the
**session** — the applier and the genetizer, with the world exactly as it was.

| Button | Effect |
|---|---|
| Start | Builds a fresh session and runs it |
| Stop | Ends the loop after the current epoch; the session stays |
| Resume | Keeps evolving the same world, epoch numbering continuing |
| Restart | What Start becomes once a session exists: throws the world away and reseeds |

Editing anything that feeds a run — a point, a variable name, a config field —
calls `markSessionDirty()`, which disables Resume. The old world was ranked
against the old data, so continuing it would silently mix the two.

`epochs` and `patience` are not shown in the form. They belong to the CLI, where
a run has to end on its own; here Stop is the answer. They still round-trip
through `getConfig()`, so a config file stays valid for both front ends.

## Threading

The contract is the one in [docs/gui.md](../../docs/gui.md): every controller
member is written on the GUI thread only. This mode adds one wrinkle — the
session has to outlive the worker without ever being touched by both.

The session is moved into a `shared_ptr` before the thread starts and moved back
into the member through `QMetaObject::invokeMethod` when the worker exits. While
it runs, the only thing crossing is `std::atomic<bool> stopRequested`. The
applier and the genetizer sit behind `unique_ptr`s inside the session because
the genetizer's rank/mutate/crossover functions capture the applier's *address*:
moving the session must not move the objects.

Snapshots carry copies, never borrows. `CollectDistinct` hands back pointers into
a world that keeps changing, so the worker copies the expression string and the
`Expression` itself before publishing. The GUI thread then evaluates those copies
to draw the curves — `Expression::run` is const and each curve owns its own
`VariableHolder`, so nothing is shared with the worker.

## What a repaint costs

The plot is immediate-mode: no `QChart`, no series objects, just a `paintEvent`.
Two things keep that affordable.

Curves are **compiled once** per snapshot rather than interpreted once per pixel
(see [`core/lib/rpn_compile.h`](../../core/lib/rpn_compile.h)), and the sampled
values are cached against the curve set and the horizontal window.

More importantly, the **finished picture is cached**. Rasterising ten
antialiased polylines costs far more than sampling them did — about 9 ms of a
13 ms frame — and hovering a point changes none of it. So the background, grid,
curves and resting points are drawn into a `QPixmap` that is invalidated only by
the view, the curve set, the points, the size or the palette, and each frame
blits that and draws just the highlight on top. A hover repaint went from 13.3 ms
to 0.95 ms, with the rendered result unchanged pixel for pixel.

Text is deliberately **not** in the cached layer: glyphs drawn into an offscreen
surface lose the subpixel antialiasing they get on a widget. Axis labels and the
legend are redrawn every frame, which is what keeps the output identical.

Dragging a point emits `FunctionsPointsModel::rowChanged` rather than `changed()`,
so only that row of the table is rewritten; `changed()` is for structural edits,
where the whole table really does have to be rebuilt.

`--print-top 0` means "all", and on the CLI that is reasonable. Here the log
goes into a text widget, so `PrintWorld` for the terminal is capped at a couple
of hundred rows. The snapshot feeding the table and the plot is a separate path
and was already bounded.

## Traps

- **`SeedThreadRng` seeds a `thread_local` engine.** It has to be called from the
  worker thread, which is why seeding happens inside the run lambda and not
  where the config is read.
- **`resetExpected()` wipes the mutation options.** The worker therefore builds a
  fresh applier per Restart and calls `setMutationOptions` / `setFitnessOptions`
  *before* `addExpected`, exactly as `cli/functions/commands.cpp` does.
- **An empty points table is not a valid run.** With no points the mutation config
  has no variables and a random organism indexes an empty list, so `start()`
  refuses before touching the worker.
- **Never push values into the form from `updateInfo()`.** `setValue` emits
  `valueChanged`, which would emit `infoUpdated`, which would call `setValue`.
  The form is only ever written at a click — see the Load config handler.
- **`QTableWidget::cellChanged` fires for programmatic writes too**, so the
  refresh raises `refreshingTable` around them; without it the table feeds its
  own values back into the model.
- **The plot never rescales itself.** Curves adapt to the user's framing, and only
  "Fit view", a CSV import, or the first show move the view.

## Testing

There is no GUI test framework here either. Behaviour is verified by hand, or by
an out-of-tree harness driving the controller with a `std::ostringstream` logger
under `QT_QPA_PLATFORM=offscreen` — the `Info` snapshot and
`setLogger(std::ostream*)` are what make the controller drivable without its
widget. The plot's gestures can be driven the same way, by sending
`QMouseEvent`s to it and checking the points table.
