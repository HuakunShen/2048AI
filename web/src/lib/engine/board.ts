/**
 * 2048 game engine — a faithful TypeScript port of the Python row-LUT engine
 * (`src/game/model/vectorboard.py` + `staticboardImpl.py`).
 *
 * A board is a `Uint8Array(16)` of **exponents** in row-major order (index =
 * row*4 + col): 0 = empty, k = tile 2^k. A single move decomposes into four
 * independent length-4 line collapses, and a line of 4 exponents (each 0..15)
 * has only 16^4 = 65536 states — so we precompute the collapse-toward-index-0
 * ("LEFT" primitive) transition tables once and every move becomes table lookups.
 *
 * The tables are built by porting `collapse_array(reverse=True)` exactly, so this
 * reproduces the Python engine's slide/merge/scoring bit-for-bit (verified by the
 * golden-parity test).
 */

export type Dir = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';

/** Directions in the exact order of Python's `ARROW_KEYS` — matters for argmax tie-breaks. */
export const DIRS: readonly Dir[] = ['UP', 'DOWN', 'LEFT', 'RIGHT'] as const;

export type Board = Uint8Array; // length 16, exponents 0..15

const GRID = 4;
const N_CELLS = GRID * GRID;
const N_ROW_STATES = 16 ** GRID; // 65536

// --------------------------------------------------------------------------- //
// Row-LUT construction (port of vectorboard._build_row_luts + collapse_array).
// --------------------------------------------------------------------------- //

/**
 * Collapse a length-4 tile-value row toward index 0 (the LEFT/UP primitive,
 * `reverse=True`), mutating `arr`. Returns the merge score and whether it moved.
 * Direct port of `NumpyStaticBoard.collapse_array(arr, reverse=True)`.
 */
function collapseRowLeft(arr: number[]): { score: number; changed: boolean } {
	const n = arr.length;
	let changed = false;
	let score = 0;
	const hasMerged = new Uint8Array(n);
	for (let i = 1; i < n; i++) {
		let curr = i;
		for (let next = i - 1; next >= 0; next--) {
			if (arr[next] === 0 && arr[curr] !== 0) {
				arr[next] = arr[curr];
				arr[curr] = 0;
				curr = next;
				changed = true;
			} else if (arr[curr] === arr[next] && !hasMerged[next] && arr[curr] !== 0) {
				arr[next] *= 2;
				arr[curr] = 0;
				hasMerged[next] = 1;
				score += arr[next];
				changed = true;
				break;
			} else {
				break;
			}
		}
	}
	return { score, changed };
}

// ROW_RESULT holds the collapsed row re-packed base-16 (Int32 to tolerate the
// pathological 2^16 overflow exactly like numpy; nibbles are masked on read).
const ROW_RESULT = new Int32Array(N_ROW_STATES);
const ROW_REWARD = new Int32Array(N_ROW_STATES);
const ROW_CHANGED = new Uint8Array(N_ROW_STATES);

(function buildRowLuts() {
	const tiles = new Array<number>(GRID);
	for (let index = 0; index < N_ROW_STATES; index++) {
		for (let c = 0; c < GRID; c++) {
			const e = (index >> (4 * c)) & 0xf;
			tiles[c] = e === 0 ? 0 : 1 << e;
		}
		const { score, changed } = collapseRowLeft(tiles);
		let outIndex = 0;
		for (let c = 0; c < GRID; c++) {
			const v = tiles[c];
			const e = v === 0 ? 0 : Math.round(Math.log2(v));
			outIndex += e << (4 * c); // unmasked, matching numpy's out_index
		}
		ROW_RESULT[index] = outIndex;
		ROW_REWARD[index] = score;
		ROW_CHANGED[index] = changed ? 1 : 0;
	}
})();

// --------------------------------------------------------------------------- //
// Line layouts: for each direction, the 4 lines, each as 4 board indices in
// collapse order (position 0 = the edge the tiles slide toward).
// --------------------------------------------------------------------------- //
const LINES: Record<Dir, number[][]> = (() => {
	const idx = (r: number, c: number) => r * GRID + c;
	const up: number[][] = [];
	const down: number[][] = [];
	const left: number[][] = [];
	const right: number[][] = [];
	for (let k = 0; k < GRID; k++) {
		left.push([idx(k, 0), idx(k, 1), idx(k, 2), idx(k, 3)]);
		right.push([idx(k, 3), idx(k, 2), idx(k, 1), idx(k, 0)]);
		up.push([idx(0, k), idx(1, k), idx(2, k), idx(3, k)]);
		down.push([idx(3, k), idx(2, k), idx(1, k), idx(0, k)]);
	}
	return { UP: up, DOWN: down, LEFT: left, RIGHT: right };
})();

export interface MoveResult {
	after: Board;
	reward: number;
	changed: boolean;
}

/**
 * Apply `dir` to `board`, returning the afterstate (no tile spawned), the merge
 * reward, and whether the board changed. Does not mutate `board`. Equivalent to
 * `NumpyStaticBoard.move(board, dir, inplace=False)`.
 */
export function move(board: Board, dir: Dir): MoveResult {
	const after = board.slice() as Board;
	let reward = 0;
	let changed = false;
	for (const line of LINES[dir]) {
		const [a, b, c, d] = line;
		const idx = board[a] | (board[b] << 4) | (board[c] << 8) | (board[d] << 12);
		const res = ROW_RESULT[idx];
		reward += ROW_REWARD[idx];
		if (ROW_CHANGED[idx]) changed = true;
		after[a] = res & 0xf;
		after[b] = (res >> 4) & 0xf;
		after[c] = (res >> 8) & 0xf;
		after[d] = (res >> 12) & 0xf;
	}
	return { after, reward, changed };
}

/** An RNG returning a float in [0, 1). Defaults to `Math.random`. */
export type Rng = () => number;

/**
 * Spawn one tile on a random empty cell: exponent 1 (tile 2) w.p. 0.9, exponent 2
 * (tile 4) w.p. 0.1. Mutates `board`; returns false if the board was full. The
 * spawn *probabilities* match the Python engine; the RNG stream need not.
 */
export function spawn(board: Board, rng: Rng = Math.random): boolean {
	let count = 0;
	for (let i = 0; i < N_CELLS; i++) if (board[i] === 0) count++;
	if (count === 0) return false;
	let pick = Math.floor(rng() * count);
	for (let i = 0; i < N_CELLS; i++) {
		if (board[i] === 0 && pick-- === 0) {
			board[i] = rng() < 0.1 ? 2 : 1;
			return true;
		}
	}
	return false;
}

/** Fresh board with two tile-2s (exponent 1) on distinct random cells, per `get_init_matrix`. */
export function initBoard(rng: Rng = Math.random): Board {
	const b = new Uint8Array(N_CELLS) as Board;
	const a = Math.floor(rng() * N_CELLS);
	b[a] = 1;
	let c = Math.floor(rng() * (N_CELLS - 1));
	if (c >= a) c++; // uniform over the remaining 15 cells
	b[c] = 1;
	return b;
}

/** Game over iff no empty cell and no orthogonally-adjacent equal pair. */
export function isDone(board: Board): boolean {
	for (let i = 0; i < N_CELLS; i++) if (board[i] === 0) return false;
	for (let r = 0; r < GRID; r++)
		for (let c = 0; c < GRID - 1; c++)
			if (board[r * GRID + c] === board[r * GRID + c + 1]) return false;
	for (let r = 0; r < GRID - 1; r++)
		for (let c = 0; c < GRID; c++)
			if (board[r * GRID + c] === board[(r + 1) * GRID + c]) return false;
	return true;
}

/** True if any tile reaches `goalExp` (default 11 = tile 2048). */
export function hasWon(board: Board, goalExp = 11): boolean {
	for (let i = 0; i < N_CELLS; i++) if (board[i] >= goalExp) return true;
	return false;
}

/** Largest exponent on the board. */
export function maxExp(board: Board): number {
	let m = 0;
	for (let i = 0; i < N_CELLS; i++) if (board[i] > m) m = board[i];
	return m;
}

/** Are there any legal moves' worth checking — used to detect a stuck board. */
export function anyMove(board: Board): boolean {
	for (const d of DIRS) if (move(board, d).changed) return true;
	return false;
}

// --------------------------------------------------------------------------- //
// Encoding helpers (exponent <-> tile value), used by the UI and golden tests.
// --------------------------------------------------------------------------- //
export const expToTile = (e: number): number => (e === 0 ? 0 : 2 ** e);
export const tileToExp = (v: number): number => (v === 0 ? 0 : Math.round(Math.log2(v)));

export function tilesToBoard(tiles: ArrayLike<number>): Board {
	const b = new Uint8Array(N_CELLS) as Board;
	for (let i = 0; i < N_CELLS; i++) b[i] = tileToExp(tiles[i]);
	return b;
}

export function boardToTiles(board: ArrayLike<number>): number[] {
	const out = new Array<number>(N_CELLS);
	for (let i = 0; i < N_CELLS; i++) out[i] = expToTile(board[i]);
	return out;
}
