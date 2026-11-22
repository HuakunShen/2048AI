import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { Engine, expToTile, tilesToBoard, type Board } from "../src/vendor/board";
import { Expectimax, type DepthCfg } from "../src/vendor/expectimax";
import { hasModel, loadModel, modelDir } from "../src/ai/loader";

const eng = new Engine(4, 4);
const model = hasModel() ? loadModel() : null;

// The weights are ~84 MB and uncommitted, so these are skipped rather than
// failed when the model is absent — see README. `describe.skipIf` keeps that
// explicit instead of silently passing an empty suite.
const withModel = describe.skipIf(model === null);

/**
 * The value-parity golden ships *inside the model directory*, not in this repo:
 * it records V(board) for the exact weights sitting next to it, so it stays in
 * sync with whichever model is installed. (The engine fixture also carries a
 * `value` field, but that one was generated from a different model and does not
 * describe these weights — reading it here would be comparing against the wrong
 * reference.)
 */
interface ValueGolden {
  shape: [number, number];
  tiles: number[];
  value: number;
}
function readValueGolden(): ValueGolden[] {
  const path = join(modelDir(), "golden.json");
  if (!existsSync(path)) return [];
  return JSON.parse(readFileSync(path, "utf8")) as ValueGolden[];
}

describe("model loader", () => {
  it("recognizes the packed weights shipped with this package", () => {
    expect(hasModel()).toBe(true);
    expect(existsSync(join(modelDir(), "model.pack"))).toBe(true);
  });

  it("reports absence rather than throwing when no model is installed", () => {
    expect(hasModel("/nonexistent/model/dir")).toBe(false);
    expect(loadModel("/nonexistent/model/dir")).toBe(null);
  });
});

withModel("n-tuple value function — golden parity vs the Python reference", () => {
  /**
   * `model.pack` stores int16 values times a per-table scale (0.75 … 3.02), so
   * one lookup is off by at most scale/2 and V() by at most their sum (~3.9).
   * The golden records the float32 reference, so compare within that bound
   * rather than for equality; the relative error stays around 1e-5, far below
   * the gaps between candidate moves.
   */
  const QUANTISATION_BOUND = 4;

  it("tracks V(board) for every golden board, across all six grid shapes", () => {
    const value = model!;
    const golden = readValueGolden();
    expect(golden.length).toBeGreaterThan(0);

    let maxDiff = 0;
    for (const [i, rec] of golden.entries()) {
      const [H, W] = rec.shape;
      const got = value.value(tilesToBoard(rec.tiles), H, W);
      const diff = Math.abs(got - rec.value);
      maxDiff = Math.max(maxDiff, diff);
      expect(diff, `golden ${i} (${H}x${W})`).toBeLessThanOrEqual(QUANTISATION_BOUND);
    }
    // One model really does serve every shape — that is the variable-grid claim.
    const shapes = new Set(golden.map((r) => r.shape.join("x")));
    expect(shapes.size).toBeGreaterThanOrEqual(4);
    expect(maxDiff).toBeLessThan(QUANTISATION_BOUND);
  });
});

withModel("expectimax", () => {
  it("picks a legal move on a live board", () => {
    const value = model!;
    const ai = new Expectimax(eng, (b: Board) => value.value(b, eng.H, eng.W));
    const board = eng.initBoard(() => 0.5);
    const { dir } = ai.getMove(board, { depth: 1 });
    expect(dir).not.toBe(null);
    expect(eng.move(board, dir!).changed).toBe(true);
  });

  it("returns dir=null on a dead board", () => {
    const value = model!;
    const ai = new Expectimax(eng, (b: Board) => value.value(b, eng.H, eng.W));
    const dead = tilesToBoard([2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2]);
    expect(eng.isDone(dead)).toBe(true);
    expect(ai.getMove(dead, { depth: 1 }).dir).toBe(null);
  });

  /**
   * The point of shipping the real trained agent rather than a heuristic: it
   * actually plays 2048. Upstream reaches 2048 at depth 2; depth 1 is used here
   * to keep the suite quick, so the bar is set lower than upstream's.
   */
  it("plays a full game and reaches a high tile", () => {
    const value = model!;
    const ai = new Expectimax(eng, (b: Board) => value.value(b, eng.H, eng.W));
    const cfg: DepthCfg = { depth: 1 };
    let board = eng.initBoard();
    let moves = 0;
    for (;;) {
      const { dir } = ai.getMove(board, cfg);
      if (!dir) break;
      const { after, changed } = eng.move(board, dir);
      if (!changed) break;
      eng.spawn(after);
      board = after;
      if (++moves > 20000) break;
    }
    const best = expToTile(eng.maxExp(board));
    expect(moves).toBeGreaterThan(100);
    expect(best).toBeGreaterThanOrEqual(512);
  }, 120_000);
});
