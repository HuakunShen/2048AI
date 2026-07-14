import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { buildValue, type Manifest } from './model';
import { compilePattern, coreLibrary } from './patterns';
import { tilesToBoard } from '../engine/board';

/**
 * Verifies the ported universal value function reproduces the Python
 * `UniversalNTuple.value()` (base path) for every golden board across shapes,
 * and that the TS pattern compiler produces the identical placements as Python.
 */
const dir = fileURLToPath(new URL('../../../static/model/', import.meta.url));

function loadValue() {
	const manifest = JSON.parse(readFileSync(dir + 'manifest.json', 'utf8')) as Manifest;
	const buffers = manifest.patterns.map((_, k) =>
		manifest.parts[k].map((_p, p) => {
			const raw = readFileSync(dir + `lut${k}_${p}.bin`);
			return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
		})
	);
	return buildValue(manifest, buffers);
}

interface Golden {
	shape: [number, number];
	tiles: number[];
	value: number;
}
const golden = JSON.parse(readFileSync(dir + 'golden.json', 'utf8')) as Golden[];

describe('universal value() parity vs Python', () => {
	it('reproduces base V(board) for every golden board and shape', () => {
		const value = loadValue();
		let maxDiff = 0;
		for (const rec of golden) {
			const [H, W] = rec.shape;
			const v = value.value(tilesToBoard(rec.tiles), H, W);
			maxDiff = Math.max(maxDiff, Math.abs(v - rec.value));
		}
		expect(maxDiff).toBeLessThan(1e-2);
	});
});

// --- placement compiler golden (does not need the model) ------------------- //
const placements = JSON.parse(
	readFileSync(fileURLToPath(new URL('./__fixtures__/placements.json', import.meta.url)), 'utf8')
) as Record<string, Record<string, number[][]>>;

describe('pattern compiler matches Python placements', () => {
	it('produces the identical instance set per shape/pattern', () => {
		const core = coreLibrary(16);
		for (const [shape, byPat] of Object.entries(placements)) {
			const [H, W] = shape.split('x').map(Number);
			for (const p of core) {
				const cp = compilePattern(p, H, W);
				const got: string[] = [];
				for (let i = 0; i < cp.nInstances; i++) {
					const row: number[] = [];
					for (let j = 0; j < cp.L; j++) row.push(cp.cells[i * cp.L + j]);
					got.push(row.join(','));
				}
				const want = byPat[p.id].map((r) => r.join(','));
				expect(got.slice().sort(), `${shape} ${p.id}`).toEqual(want.slice().sort());
			}
		}
	});
});
