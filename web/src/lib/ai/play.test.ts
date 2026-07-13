import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { allocLuts, makeValue, N_TUPLES, scatterInto, type ValueFn } from './ntuple';
import { Expectimax, type DepthCfg } from './expectimax';
import { initBoard, move, spawn, maxExp, expToTile, type Board } from '../engine/board';

/**
 * Full-stack AI integration: runs the exact `Expectimax` + `makeValue` code the
 * Web Worker uses (in Node, without the kkrpc plumbing) to play complete games,
 * proving the browser AI actually plays strongly — not just that V() matches.
 */
function loadValue(): ValueFn {
	const dir = fileURLToPath(new URL('../../../static/model/', import.meta.url));
	const manifest = JSON.parse(readFileSync(dir + 'manifest.json', 'utf8'));
	const luts = allocLuts();
	for (let t = 0; t < N_TUPLES; t++) {
		const nnz = manifest.counts[t];
		const raw = readFileSync(dir + `lut${t}.bin`);
		const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
		scatterInto(luts[t], new Uint32Array(ab, 0, nnz), new Float32Array(ab, nnz * 4, nnz));
	}
	return makeValue(luts);
}

function playGame(ai: Expectimax, cfg: DepthCfg): { maxTile: number; moves: number } {
	let board: Board = initBoard();
	let moves = 0;
	for (;;) {
		const { dir } = ai.getMove(board, cfg);
		if (!dir) break;
		const { after, changed } = move(board, dir);
		if (!changed) break;
		spawn(after);
		board = after;
		moves++;
		if (moves > 20000) break; // safety
	}
	return { maxTile: expToTile(maxExp(board)), moves };
}

describe('AI integration — plays real games to a high tile', () => {
	it('reaches the 2048 tile at expectimax depth 2', () => {
		const ai = new Expectimax(loadValue());
		const cfg: DepthCfg = { depth: 2 };
		let bestTile = 0;
		const results: string[] = [];
		for (let g = 0; g < 3; g++) {
			const t0 = performance.now();
			const { maxTile, moves } = playGame(ai, cfg);
			results.push(`tile=${maxTile} moves=${moves} in ${Math.round(performance.now() - t0)}ms`);
			bestTile = Math.max(bestTile, maxTile);
			if (bestTile >= 2048) break; // one win is enough to prove the pipeline
		}
		console.log('[depth-2 games]', results.join(' | '));
		// depth-2 reaches 2048 ~96% of the time, so 3 tries essentially never all fail.
		expect(bestTile, `max tiles: ${results.join(', ')}`).toBeGreaterThanOrEqual(2048);
	}, 120_000);
});
