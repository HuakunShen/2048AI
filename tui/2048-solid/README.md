# 2048AI Solid TUI

**2048 in the terminal, authored in Solid — with the real trained AI.**

The AI here is not a heuristic. It is an n-tuple network (5 patterns, 8-way
dihedral symmetry, ~84 MB of trained weights) driving an expectimax search over
afterstates — the same agent as the web app it was ported from. At depth 2 it
reaches the 4096 tile.

```bash
pnpm --dir tui/2048-solid install
pnpm --dir tui/2048-solid dev
```

| key | |
|---|---|
| `↑ ↓ ← →` / `h j k l` | move |
| `a` | toggle AI auto-play |
| `s` | one AI move |
| `+` / `-` | search depth (1–4) |
| `n` | new game |
| `q` / `Ctrl-C` | quit |

## The weights ship with the game

The trained n-tuple weights are included as one compact `model/model.pack` file
(about 32 MB). It is decoded into the dense lookup tables required by the
expectimax hot path at startup. The compact format is substantially smaller than
the original 84 MB sharded export and is easier to distribute with this package.

The package also keeps the model metadata and value golden used by the tests:

```
model/
  model.pack
  manifest.json
  golden.json        # V(board) references for these exact weights
```

For development, `UNIVIEW_2048_MODEL_DIR` can point at either this packed format
or the older sharded `manifest.json` + `lut*.bin` format:

```bash
UNIVIEW_2048_MODEL_DIR=/path/to/model pnpm --dir tui/2048-solid dev
```

## Layout

```
src/
  vendor/     engine + AI, vendored verbatim from the training repo (pure, no DOM)
              board.ts · patterns.ts · universal.ts · model.ts · expectimax.ts
  ai/
    loader.ts     reads packed weights first, then the legacy sharded format
    controller.ts auto-play state (available / running / depth / step)
  game.ts     the game controller — engine + signals, no UI, injectable RNG
  board.tsx   the grid, in the classic 2048 palette
  app.tsx     board + score + score-curve sparkline + AI panel
  keys.ts     input mapping (kept separate so it is testable without a terminal)
```

`src/vendor/` currently sits close to upstream (only an import path changed), which
makes it cheap to re-sync a fix from the training repo. That is a convenience, not
a rule — change it freely if the game needs it. The golden tests below are what
actually keeps it honest.

## Why it is trustworthy

The port is verified against the reference implementation, not eyeballed:

- **Engine** — replays all 204 golden boards × 4 moves (816 cases) from the
  Python reference; `after`, `reward` and `changed` all match.
- **Value function** — reproduces the reference `V(board)` on every golden board
  across all six grid shapes (4×4, 5×5, 4×5, 5×4, 3×4, 6×6), max diff < 1e-3.
  One model really does serve every shape.
- **It reaches 2048.** Not a claim — a test: at the depth the app ships with, a
  seeded game is played out through the real game controller and must end with
  `game.won() === true`. Depth 2 goes on to 4096.

The value-parity golden lives beside the shipped model, because it describes the
exact weights sitting beside it.

```bash
pnpm --dir tui/2048-solid test
pnpm --dir tui/2048-solid check-types
```

## What it demonstrates for uniview

Everything on screen is a `@uniview/tui-solid` component: `Panel`, `Box`, `Text`,
`StatusBar`, and the score curve is the same `Sparkline` the charts demo uses —
no bespoke rendering. The tiles are plain colored `Box`es, so no new render
primitive was needed for a game.

State is signals; the app mounts once and never re-renders by hand.

The app consumes the public `@uniview/tui-core@0.0.1` and
`@uniview/tui-solid@0.0.1` packages directly; it does not depend on Uniview's
monorepo `workspace:*` links.
