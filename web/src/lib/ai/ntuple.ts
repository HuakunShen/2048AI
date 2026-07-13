/**
 * N-tuple value function — a TypeScript port of `src/agent/ntuple.py`.
 *
 * `V(board)` is the sum of 32 lookups into 4 large tables: four overlapping
 * 6-cell tuples, each read under all 8 dihedral board symmetries (weight sharing).
 * This is NOT a neural network — no matmuls, just table gather + sum — so it is a
 * few lines of arithmetic over typed arrays. Direct port of `_value_njit`.
 */

export const INDEX_BASE = 16;
export const TUPLE_LEN = 6;
export const N_SYMS = 8;
export const TABLE_SIZE = INDEX_BASE ** TUPLE_LEN; // 16^6 = 16,777,216
export const MAX_EXPONENT = 15;

/** Four overlapping 6-cell tuples (row-major flat indices), from `ntuple.py`. */
const TUPLES: readonly number[][] = [
	[0, 1, 2, 3, 4, 5],
	[4, 5, 6, 7, 8, 9],
	[0, 1, 2, 4, 5, 6],
	[4, 5, 6, 8, 9, 10]
];
export const N_TUPLES = TUPLES.length;

// --- dihedral symmetry (port of _symmetry_perms) --------------------------- //
const fliplr = (m: number[][]): number[][] => m.map((row) => [...row].reverse());
const rot90 = (m: number[][]): number[][] => {
	// numpy rot90 counterclockwise once: out[i][j] = m[j][n-1-i]
	const n = m.length;
	const out = Array.from({ length: n }, () => new Array<number>(n));
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] = m[j][n - 1 - i];
	return out;
};

function symmetryPerms(): number[][] {
	const grid: number[][] = [
		[0, 1, 2, 3],
		[4, 5, 6, 7],
		[8, 9, 10, 11],
		[12, 13, 14, 15]
	];
	const perms: number[][] = [];
	for (const flip of [false, true]) {
		let cur = flip ? fliplr(grid) : grid;
		for (let k = 0; k < 4; k++) {
			perms.push(cur.flat());
			cur = rot90(cur);
		}
	}
	return perms; // 8 permutations of the 16 flat indices
}

// CELLS[i] = the 6 source cells for instance i (tuple i>>3, symmetry i&7),
// flattened to a typed array for the hot loop. POW packs the 6 cells base-16.
const PERMS = symmetryPerms();
const N_INSTANCES = N_TUPLES * N_SYMS; // 32
const CELLS = new Int32Array(N_INSTANCES * TUPLE_LEN);
{
	let i = 0;
	for (const t of TUPLES) {
		for (const perm of PERMS) {
			for (let c = 0; c < TUPLE_LEN; c++) CELLS[i * TUPLE_LEN + c] = perm[t[c]];
			i++;
		}
	}
}
const POW = new Int32Array([1, 16, 256, 4096, 65536, 1048576]); // 16^[0..5]

export type Luts = Float32Array[]; // length N_TUPLES, each TABLE_SIZE

export type ValueFn = (board: ArrayLike<number>) => number;

/**
 * Build `value(board)` bound to the given lookup tables. `board` is a 16-cell
 * exponent array (0..15). Matches `_value_njit` exactly (float64 accumulation).
 */
export function makeValue(luts: Luts): ValueFn {
	return function value(board: ArrayLike<number>): number {
		let total = 0;
		for (let inst = 0; inst < N_INSTANCES; inst++) {
			const base = inst * TUPLE_LEN;
			let idx = 0;
			for (let c = 0; c < TUPLE_LEN; c++) {
				let e = board[CELLS[base + c]];
				if (e > MAX_EXPONENT) e = MAX_EXPONENT;
				idx += e * POW[c];
			}
			total += luts[inst >> 3][idx]; // inst >> 3 == inst / N_SYMS
		}
		return total;
	};
}

/** Allocate the 4 dense zero-filled tables (~256 MB total). */
export function allocLuts(): Luts {
	const luts: Float32Array[] = [];
	for (let t = 0; t < N_TUPLES; t++) luts.push(new Float32Array(TABLE_SIZE));
	return luts;
}

/** Scatter sparse `(index, value)` pairs into a dense table: `lut[index[i]] = value[i]`. */
export function scatterInto(lut: Float32Array, indices: Uint32Array, values: Float32Array): void {
	for (let i = 0; i < indices.length; i++) lut[indices[i]] = values[i];
}
