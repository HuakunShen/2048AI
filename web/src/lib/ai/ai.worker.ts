/**
 * AI Web Worker — holds the shared universal n-tuple tables and runs expectimax
 * off the UI thread, for **any board shape**. Exposed as a typed API over kkrpc.
 *
 * The main thread calls `init(base, onProgress)` once (downloads + reconstructs
 * the per-pattern tables) then `bestMove(board, H, W, cfg)` per move. The value
 * function is shape-agnostic; the worker compiles placements per shape on demand.
 */
/// <reference lib="webworker" />
import { expose } from 'kkrpc';
import { workerSelfTransport } from 'kkrpc/worker';
import { buildValue, type Manifest } from './model';
import { UniversalValue } from './universal';
import { Expectimax, type DepthCfg } from './expectimax';
import { Engine, type Board, type Dir } from '../engine/board';

let value: UniversalValue | null = null;
const searchers = new Map<string, Expectimax>();

let modelBase = '/model/';
const modelUrl = (name: string) => new URL(name, modelBase).href;

function searcherFor(H: number, W: number): Expectimax {
	const key = `${H}x${W}`;
	let s = searchers.get(key);
	if (!s) {
		const eng = new Engine(H, W);
		s = new Expectimax(eng, (b: Board) => value!.value(b, H, W));
		searchers.set(key, s);
	}
	return s;
}

const aiApi = {
	/** Download + reconstruct the shared tables. `onProgress(0..1)` fires per table. */
	async init(base: string, onProgress?: (fraction: number) => void): Promise<void> {
		modelBase = base;
		if (value) {
			onProgress?.(1);
			return;
		}
		const res = await fetch(modelUrl('manifest.json'));
		if (!res.ok) throw new Error(`failed to fetch manifest.json: ${res.status}`);
		const manifest = (await res.json()) as Manifest;

		const totalParts = manifest.parts.reduce((a, ps) => a + ps.length, 0);
		let done = 0;
		const buffers: ArrayBuffer[][] = [];
		for (let k = 0; k < manifest.patterns.length; k++) {
			const partBufs: ArrayBuffer[] = [];
			for (let p = 0; p < manifest.parts[k].length; p++) {
				const r = await fetch(modelUrl(`lut${k}_${p}.bin`));
				if (!r.ok) throw new Error(`failed to fetch lut${k}_${p}.bin: ${r.status}`);
				partBufs.push(await r.arrayBuffer());
				onProgress?.(++done / totalParts);
			}
			buffers.push(partBufs);
		}
		value = buildValue(manifest, buffers);
		onProgress?.(1);
	},

	/** Best direction for an `H*W` exponent `board`, or dir=null if stuck. */
	async bestMove(
		board: number[],
		H: number,
		W: number,
		cfg: DepthCfg
	): Promise<{ dir: Dir | null; value: number }> {
		if (!value) throw new Error('AI not initialized — call init() first');
		return searcherFor(H, W).getMove(Uint8Array.from(board) as Board, cfg);
	}
};

export type AiAPI = typeof aiApi;

expose(aiApi, workerSelfTransport());
