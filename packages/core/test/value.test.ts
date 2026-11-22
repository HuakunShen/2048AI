import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { compilePattern, coreLibrary } from "../src/ai/patterns";
import { tilesToBoard } from "../src/engine";
import { loadValue, quantisationBound } from "./load-model";

/**
 * Verifies the ported universal value function still tracks the Python
 * `UniversalNTuple.value()` reference for every golden board across shapes, and
 * that the TS pattern compiler produces the identical placements as Python.
 *
 * The golden records the **float32** reference; `assets/model.pack` stores int16
 * values times a per-table scale, so equality is not the right assertion. The
 * tolerance is derived from the pack's own scales ({@link quantisationBound}),
 * which makes this a check on the decoder rather than a tautology against itself.
 */
interface Golden {
	shape: [number, number];
	tiles: number[];
	value: number;
}
const golden = JSON.parse(
	readFileSync(fileURLToPath(new URL("../../../assets/golden.json", import.meta.url)), "utf8")
) as Golden[];

describe("universal value() parity vs Python", () => {
	it("tracks base V(board) for every golden board and shape within the quantisation bound", () => {
		const value = loadValue();
		const bound = quantisationBound();
		let maxDiff = 0;
		let maxRelative = 0;
		for (const rec of golden) {
			const [H, W] = rec.shape;
			const v = value.value(tilesToBoard(rec.tiles), H, W);
			const diff = Math.abs(v - rec.value);
			maxDiff = Math.max(maxDiff, diff);
			maxRelative = Math.max(maxRelative, diff / Math.max(1, Math.abs(rec.value)));
		}
		console.log(
			`[value] maxDiff=${maxDiff.toFixed(4)} bound=${bound.toFixed(4)} maxRelative=${maxRelative.toExponential(2)}`
		);
		expect(maxDiff).toBeLessThanOrEqual(bound + 1e-6);
		// Ordering is what expectimax consumes; a few parts in 10^5 is far below
		// the gaps between moves, so the quantised pack is not a weaker player.
		expect(maxRelative).toBeLessThan(1e-4);
	});
});

// --- placement compiler golden (does not need the model) ------------------- //
const placements = JSON.parse(
	readFileSync(fileURLToPath(new URL("./fixtures/placements.json", import.meta.url)), "utf8")
) as Record<string, Record<string, number[][]>>;

describe("pattern compiler matches Python placements", () => {
	it("produces the identical instance set per shape/pattern", () => {
		const core = coreLibrary(16);
		for (const [shape, byPat] of Object.entries(placements)) {
			const [H, W] = shape.split("x").map(Number);
			for (const p of core) {
				const cp = compilePattern(p, H!, W!);
				const got: string[] = [];
				for (let i = 0; i < cp.nInstances; i++) {
					const row: number[] = [];
					for (let j = 0; j < cp.L; j++) row.push(cp.cells[i * cp.L + j]!);
					got.push(row.join(","));
				}
				const want = byPat[p.id]!.map((r) => r.join(","));
				expect(got.slice().sort(), `${shape} ${p.id}`).toEqual(want.slice().sort());
			}
		}
	});
});
