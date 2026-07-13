/**
 * Expectimax search over afterstates with an n-tuple leaf evaluator — a port of
 * `src/agent/expectimax.py`.
 *
 * The tree alternates MAX nodes (our move) and CHANCE nodes (the random tile
 * spawn: 2 w.p. 0.9, 4 w.p. 0.1). `depth == 1` reduces to the greedy n-tuple
 * player, `a* = argmax_a [reward(s,a) + V(afterstate)]`. A transposition table
 * memoizes chance-node values within a single root search, and adaptive depth
 * searches deeper as the board fills.
 */
import { DIRS, move, type Board, type Dir } from '../engine/board';
import type { ValueFn } from './ntuple';

/** Spawned tile as exponent + probability: 2 (exp 1) w.p. .9, 4 (exp 2) w.p. .1. */
const SPAWN_TILES: readonly [number, number][] = [
	[1, 0.9],
	[2, 0.1]
];

export interface DepthCfg {
	/** Base search depth in move plies; 1 == greedy. */
	depth: number;
	/** Optional `[maxEmpty, depth]` rules (first match wins) overriding `depth` per move. */
	adaptive?: [number, number][];
	/** Expand at most this many empty cells at a chance node (default 16 ≥ 15, so full). */
	maxChanceCells?: number;
}

function keyOf(board: Board, depth: number): string {
	let s = '';
	for (let i = 0; i < 16; i++) s += board[i].toString(16); // 16 nibbles, 0..f
	return s + depth;
}

export class Expectimax {
	private readonly value: ValueFn;
	private tt = new Map<string, number>();

	constructor(value: ValueFn) {
		this.value = value;
	}

	private effectiveDepth(board: Board, cfg: DepthCfg): number {
		if (!cfg.adaptive || cfg.adaptive.length === 0) return cfg.depth;
		let nEmpty = 0;
		for (let i = 0; i < 16; i++) if (board[i] === 0) nEmpty++;
		for (const [maxEmpty, d] of cfg.adaptive) if (nEmpty <= maxEmpty) return d;
		return cfg.adaptive[cfg.adaptive.length - 1][1];
	}

	/** Expected value of an afterstate: leaf V at depth 0, else a chance node. */
	private evaluateAfterstate(after: Board, depth: number, maxChance: number): number {
		if (depth <= 0) return this.value(after);

		const key = keyOf(after, depth);
		const cached = this.tt.get(key);
		if (cached !== undefined) return cached;

		const empties: number[] = [];
		for (let i = 0; i < 16; i++) if (after[i] === 0) empties.push(i);
		if (empties.length === 0) {
			const v = this.value(after);
			this.tt.set(key, v);
			return v;
		}

		// maxChance defaults to 16 ≥ the 15-cell max, so the full expansion always
		// runs (matching the Python default). The slice is a deterministic fallback.
		const cells = empties.length > maxChance ? empties.slice(0, maxChance) : empties;
		let total = 0;
		for (const cell of cells) {
			for (const [tile, p] of SPAWN_TILES) {
				after[cell] = tile;
				total += p * this.bestActionValue(after, depth, maxChance);
				after[cell] = 0;
			}
		}
		const v = total / cells.length;
		this.tt.set(key, v);
		return v;
	}

	/** MAX node: best over valid moves of reward + deeper afterstate value. */
	private bestActionValue(state: Board, depth: number, maxChance: number): number {
		let best = -Infinity;
		for (const d of DIRS) {
			const { after, reward, changed } = move(state, d);
			if (!changed) continue;
			const val = reward + this.evaluateAfterstate(after, depth - 1, maxChance);
			if (val > best) best = val;
		}
		return best === -Infinity ? 0 : best;
	}

	/** Root MAX node: return the best direction (and its value), or `null` if stuck. */
	getMove(board: Board, cfg: DepthCfg): { dir: Dir | null; value: number } {
		this.tt.clear();
		const depth = this.effectiveDepth(board, cfg);
		const maxChance = cfg.maxChanceCells ?? 16;
		let bestDir: Dir | null = null;
		let bestVal = -Infinity;
		for (const d of DIRS) {
			const { after, reward, changed } = move(board, d);
			if (!changed) continue;
			const val = reward + this.evaluateAfterstate(after, depth - 1, maxChance);
			if (val > bestVal) {
				bestVal = val;
				bestDir = d;
			}
		}
		return { dir: bestDir, value: bestVal === -Infinity ? 0 : bestVal };
	}
}
