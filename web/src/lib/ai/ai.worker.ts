/**
 * AI Web Worker — holds the ~256MB dense n-tuple tables and runs expectimax off
 * the UI thread. Exposed as a typed API over kkrpc (`kkrpc/worker` transport).
 *
 * The main thread calls `init(onProgress)` once (downloads + scatters the weights)
 * then `bestMove(board, cfg)` per move. Only the tiny board/direction cross the
 * RPC — the 256MB tables never do; the worker fetches them itself.
 */
/// <reference lib="webworker" />
import { expose } from 'kkrpc';
import { workerSelfTransport } from 'kkrpc/worker';
import { allocLuts, makeValue, scatterInto, type Luts } from './ntuple';
import { Expectimax, type DepthCfg } from './expectimax';
import type { Dir } from '../engine/board';

interface Manifest {
	tableSize: number;
	nTuples: number;
	counts: number[];
}

let searcher: Expectimax | null = null;

// Set by init() to an absolute `.../model/` URL. The main thread resolves it
// (base-path aware) and passes it in, because inside the bundled worker
// `import.meta.env.BASE_URL` is relative to the worker's own chunk path, which
// in production (`/_app/immutable/workers/…`) points at the wrong directory.
let modelBase = '/model/';
const modelUrl = (name: string) => new URL(name, modelBase).href;

async function loadTable(luts: Luts, t: number, nnz: number): Promise<void> {
	const res = await fetch(modelUrl(`lut${t}.bin`));
	if (!res.ok) throw new Error(`failed to fetch lut${t}.bin: ${res.status}`);
	const buf = await res.arrayBuffer();
	const indices = new Uint32Array(buf, 0, nnz);
	const values = new Float32Array(buf, nnz * 4, nnz);
	scatterInto(luts[t], indices, values);
}

const aiApi = {
	/**
	 * Download + scatter the weights and build the searcher. `base` is an absolute
	 * `.../model/` URL (from the main thread); `onProgress(0..1)` fires per table.
	 */
	async init(base: string, onProgress?: (fraction: number) => void): Promise<void> {
		modelBase = base;
		if (searcher) {
			onProgress?.(1);
			return;
		}
		const res = await fetch(modelUrl('manifest.json'));
		if (!res.ok) throw new Error(`failed to fetch manifest.json: ${res.status}`);
		const manifest = (await res.json()) as Manifest;

		const luts = allocLuts();
		const totalNnz = manifest.counts.reduce((a, b) => a + b, 0);
		let done = 0;
		for (let t = 0; t < manifest.nTuples; t++) {
			await loadTable(luts, t, manifest.counts[t]);
			done += manifest.counts[t];
			onProgress?.(done / totalNnz);
		}
		searcher = new Expectimax(makeValue(luts));
		onProgress?.(1);
	},

	/** Best direction (and its value) for a 16-cell exponent `board`, or dir=null if stuck. */
	async bestMove(board: number[], cfg: DepthCfg): Promise<{ dir: Dir | null; value: number }> {
		if (!searcher) throw new Error('AI not initialized — call init() first');
		return searcher.getMove(Uint8Array.from(board), cfg);
	}
};

export type AiAPI = typeof aiApi;

expose(aiApi, workerSelfTransport());
