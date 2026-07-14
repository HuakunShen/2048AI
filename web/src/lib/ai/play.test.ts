import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { buildValue, type Manifest } from './model';
import { Expectimax, type DepthCfg } from './expectimax';
import { Engine, expToTile, type Board } from '../engine/board';
import type { UniversalValue } from './universal';

/**
 * Full-stack AI integration: runs the exact `Engine` + `UniversalValue` +
 * `Expectimax` code the Web Worker uses (in Node) to play complete games on
 * several board shapes with one model — proving the variable-grid browser AI
 * actually plays, not just that V() matches.
 */
const dir = fileURLToPath(new URL('../../../static/model/', import.meta.url));

function loadValue(): UniversalValue {
	const manifest = JSON.parse(readFileSync(dir + 'manifest.json', 'utf8')) as Manifest;
	const buffers = manifest.patterns.map((_, k) =>
		manifest.parts[k].map((_p, p) => {
			const raw = readFileSync(dir + `lut${k}_${p}.bin`);
			return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
		})
	);
	return buildValue(manifest, buffers);
}

function playGame(eng: Engine, value: UniversalValue, cfg: DepthCfg) {
	const ai = new Expectimax(eng, (b: Board) => value.value(b, eng.H, eng.W));
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
	return { maxTile: expToTile(eng.maxExp(board)), moves };
}

describe('variable-grid AI integration — one model plays multiple shapes', () => {
	const value = loadValue();
	const cfg: DepthCfg = { depth: 2 };

	it('reaches 2048 on 4x4 (depth-2)', () => {
		const eng = new Engine(4, 4);
		let best = 0;
		const log: string[] = [];
		for (let g = 0; g < 3 && best < 2048; g++) {
			const { maxTile, moves } = playGame(eng, value, cfg);
			log.push(`tile=${maxTile} moves=${moves}`);
			best = Math.max(best, maxTile);
		}
		console.log('[4x4 depth-2]', log.join(' | '));
		expect(best, log.join(', ')).toBeGreaterThanOrEqual(2048);
	}, 120_000);

	it('plays a full 5x5 game and reaches a high tile', () => {
		const eng = new Engine(5, 5);
		const { maxTile, moves } = playGame(eng, value, cfg);
		console.log(`[5x5 depth-2] tile=${maxTile} moves=${moves}`);
		expect(moves).toBeGreaterThan(50);
		expect(maxTile).toBeGreaterThanOrEqual(2048);
	}, 120_000);

	it('plays a non-square 3x4 game to completion', () => {
		const eng = new Engine(3, 4);
		const { maxTile, moves } = playGame(eng, value, cfg);
		console.log(`[3x4 depth-2] tile=${maxTile} moves=${moves}`);
		expect(moves).toBeGreaterThan(10);
		expect(maxTile).toBeGreaterThanOrEqual(256);
	}, 120_000);
});
