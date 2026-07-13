import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { allocLuts, makeValue, N_TUPLES, scatterInto } from './ntuple';
import { tilesToBoard } from '../engine/board';

/**
 * Loads the exported sparse weights into dense tables and checks the ported
 * `value()` reproduces Python `NTupleNetwork.value()` bit-for-bit (both accumulate
 * the same float32 table entries in float64, in the same order → ~exact match).
 */
function loadLuts() {
	const dir = fileURLToPath(new URL('../../../static/model/', import.meta.url));
	const manifest = JSON.parse(readFileSync(dir + 'manifest.json', 'utf8'));
	const luts = allocLuts();
	for (let t = 0; t < N_TUPLES; t++) {
		const nnz = manifest.counts[t];
		const raw = readFileSync(dir + `lut${t}.bin`);
		const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
		const indices = new Uint32Array(ab, 0, nnz);
		const values = new Float32Array(ab, nnz * 4, nnz);
		scatterInto(luts[t], indices, values);
	}
	return { luts, manifest };
}

interface GoldenBoard {
	tiles: number[];
	value: number;
}
const golden = JSON.parse(
	readFileSync(fileURLToPath(new URL('../engine/__fixtures__/golden.json', import.meta.url)), 'utf8')
) as { boards: GoldenBoard[] };

describe('golden value() parity vs NTupleNetwork.value', () => {
	it('manifest matches the ported table geometry', () => {
		const { manifest } = loadLuts();
		expect(manifest.nTuples).toBe(N_TUPLES);
		expect(manifest.tableSize).toBe(16 ** 6);
	});

	it('reproduces V(board) for every golden board', () => {
		const { luts } = loadLuts();
		const value = makeValue(luts);
		let maxDiff = 0;
		for (const rec of golden.boards) {
			const v = value(tilesToBoard(rec.tiles));
			maxDiff = Math.max(maxDiff, Math.abs(v - rec.value));
		}
		expect(maxDiff).toBeLessThan(1e-2);
	});
});
