/**
 * Main-thread handle to the AI worker. `createAi()` spins up the worker and
 * returns its typed API (via kkrpc `wrap`), so the UI can `await ai.init(...)`
 * and `await ai.bestMove(...)` as if they were local async functions.
 */
import { wrap } from 'kkrpc';
import { workerTransport } from 'kkrpc/worker';
import AiWorker from './ai.worker?worker';
import type { AiAPI } from './ai.worker';

export type { AiAPI };
export type { DepthCfg } from './expectimax';

export function createAi(): AiAPI {
	const worker = new AiWorker();
	return wrap<AiAPI>(workerTransport(worker));
}
