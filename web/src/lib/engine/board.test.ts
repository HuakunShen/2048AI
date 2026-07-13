import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
	boardToTiles,
	DIRS,
	initBoard,
	isDone,
	move,
	spawn,
	tilesToBoard,
	type Board,
	type Dir
} from './board';

/** Build an exponent board from a 4x4 matrix of tile values. */
function fromTiles(rows: number[][]): Board {
	return tilesToBoard(rows.flat());
}

describe('collapse / move semantics', () => {
	it('slides and merges a row leftward', () => {
		const { after, reward, changed } = move(fromTiles([[2, 2, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 'LEFT');
		expect(boardToTiles(after).slice(0, 4)).toEqual([4, 4, 0, 0]);
		expect(reward).toBe(4);
		expect(changed).toBe(true);
	});

	it('merges four equal tiles into two (no double-merge)', () => {
		const { after, reward } = move(fromTiles([[4, 4, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), 'LEFT');
		expect(boardToTiles(after).slice(0, 4)).toEqual([8, 8, 0, 0]);
		expect(reward).toBe(16);
	});

	it('collapses columns upward', () => {
		const { after } = move(fromTiles([[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]]), 'UP');
		expect(boardToTiles(after).filter((_, i) => i % 4 === 0)).toEqual([4, 8, 0, 0]);
	});

	it('reports changed=false for a no-op move', () => {
		const { changed, reward } = move(fromTiles([[4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4]]), 'LEFT');
		expect(changed).toBe(false);
		expect(reward).toBe(0);
	});
});

describe('board lifecycle', () => {
	it('initBoard places exactly two tile-2s', () => {
		const b = initBoard(() => 0.5);
		const tiles = boardToTiles(b);
		expect(tiles.filter((t) => t === 2)).toHaveLength(2);
		expect(tiles.filter((t) => t === 0)).toHaveLength(14);
	});

	it('spawn fills exactly one empty cell with 2 or 4', () => {
		const b = fromTiles([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [2, 4, 8, 0]]);
		const ok = spawn(b, () => 0.99); // >= 0.1 → tile 2
		expect(ok).toBe(true);
		expect(boardToTiles(b)[15]).toBe(2);
	});

	it('isDone is true only for a full board with no merges', () => {
		expect(isDone(fromTiles([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]))).toBe(true);
		expect(isDone(fromTiles([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]]))).toBe(false);
		expect(isDone(fromTiles([[2, 2, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 8]]))).toBe(false);
	});
});

interface GoldenBoard {
	tiles: number[];
	value: number;
	moves: Record<Dir, { after: number[]; reward: number; changed: boolean }>;
}
const golden = JSON.parse(
	readFileSync(fileURLToPath(new URL('./__fixtures__/golden.json', import.meta.url)), 'utf8')
) as { boards: GoldenBoard[] };

describe('golden engine parity vs NumpyStaticBoard', () => {
	it('reproduces afterstate, reward and changed for every recorded board/direction', () => {
		let checks = 0;
		for (const rec of golden.boards) {
			const b = tilesToBoard(rec.tiles);
			for (const d of DIRS) {
				const { after, reward, changed } = move(b, d);
				const exp = rec.moves[d];
				expect(boardToTiles(after), `after ${d} on ${rec.tiles}`).toEqual(exp.after);
				expect(reward, `reward ${d}`).toBe(exp.reward);
				expect(changed, `changed ${d}`).toBe(exp.changed);
				checks++;
			}
		}
		expect(checks).toBe(golden.boards.length * 4);
	});
});
