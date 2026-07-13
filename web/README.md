# 2048 AI — client-side web app

Plays 2048 and runs the trained **n-tuple + expectimax** AI **entirely in the browser** (no AI
server). SvelteKit + `@sveltejs/adapter-cloudflare`, deployed to Cloudflare Workers.

## How it works

- **Engine + AI ported to TypeScript** (`src/lib/engine/board.ts`, `src/lib/ai/`): the row-LUT game
  engine, the n-tuple value function, and expectimax search — a faithful port of the Python
  originals (`src/agent/ntuple.py`, `src/agent/expectimax.py`). The n-tuple is a lookup-table value
  function, **not** a neural net, so there is no ONNX/torch — just typed-array arithmetic.
- **Web Worker** (`src/lib/ai/ai.worker.ts`) holds the ~256 MB dense weight tables and runs the
  search off the UI thread. It's exposed to the page as a typed async API over
  [**kkrpc**](https://github.com/kunkunsh/kkrpc) (`wrap`/`expose` + `kkrpc/worker`).
- **Weights are shipped as static assets.** The trained tables are ~7% non-zero, exported to
  `static/model/lut{0..3}.bin` (uint32 index + float32 value) + `manifest.json` — ~22 MB gzipped,
  fetched + scattered into dense `Float32Array`s in the worker on first load.
- **UI** uses [`@kksh/svelte5`](https://www.npmjs.com/package/@kksh/svelte5) (Svelte 5 + Tailwind v4):
  human play (arrow keys / swipe), AI auto-play with a speed slider, hint/assist mode, dark/light.

## Generate the model weights (required before build)

The weights are **gitignored** and regenerated from the trained checkpoint. From the **repo root**:

```sh
uv run scripts/export_web_model.py     # -> web/static/model/{lut0..3.bin, manifest.json}
uv run scripts/dump_golden.py          # -> src/lib/engine/__fixtures__/golden.json (test fixture)
```

## Develop / test / build

```sh
bun install
bun run dev                 # http://localhost:5173
bun run test:unit -- --run  # vitest: engine+value golden parity vs Python, AI plays to 2048
bunx playwright test        # e2e: boots the worker + AI in a real browser and auto-plays
bun run build               # vite build -> .svelte-kit/cloudflare (adapter-cloudflare)
```

## Deploy to Cloudflare Workers

```sh
wrangler login              # interactive, one-time
bun run deploy              # build + wrangler deploy
```

`wrangler.jsonc` serves the prerendered shell + `static/model/*.bin` as static assets (each < 25 MiB).

## Notes

- **Desktop-first / full precision:** the dense fp32 tables use ~256 MB in the worker. First visit
  downloads ~22 MB (then browser/CDN-cached). A mobile-lean variant (fp16 + hash) is future work.
- **Search strength:** *Fast* = greedy, *Strong* = expectimax depth 2 (default, ~96% @2048),
  *Max* = depth 3 with adaptive endgame deepening (~99% @2048, slower per move).
