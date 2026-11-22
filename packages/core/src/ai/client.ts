/**
 * Main-thread handle to the AI Worker.
 *
 * The host creates the Worker (so its bundler emits the chunk) and hands it
 * here; every request is a promise keyed by a monotonically increasing id.
 * A failed `init` rejects and leaves a playable human game — callers must not
 * treat a missing model as a fatal error.
 */
import type { Dir } from "../engine";
import type { DepthCfg } from "./expectimax";
import type { AiProvider, AiMove } from "../game/controller";
import type { AiErrorCode, WorkerRequest, WorkerResponse } from "./protocol";

export class AiUnavailableError extends Error {
	readonly code: AiErrorCode;
	constructor(code: AiErrorCode, message: string) {
		super(message);
		this.name = "AiUnavailableError";
		this.code = code;
	}
}

interface Pending {
	resolve(value: AiMove | void): void;
	reject(reason: unknown): void;
	onProgress?: (fraction: number) => void;
}

export function createWorkerAi(worker: Worker): AiProvider & { dispose(): void } {
	let nextId = 1;
	const pending = new Map<number, Pending>();

	worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
		const message = event.data;
		const entry = pending.get(message.id);
		if (entry === undefined) return;
		switch (message.kind) {
			case "progress":
				entry.onProgress?.(message.fraction);
				return;
			case "ready":
				pending.delete(message.id);
				entry.resolve();
				return;
			case "error":
				pending.delete(message.id);
				entry.reject(new AiUnavailableError(message.code, message.message));
				return;
			case "move":
				pending.delete(message.id);
				entry.resolve({ dir: message.dir, value: message.value });
				return;
		}
	});

	const send = (request: WorkerRequest, onProgress?: (fraction: number) => void): Promise<AiMove | void> =>
		new Promise((resolve, reject) => {
			pending.set(request.id, { resolve, reject, ...(onProgress === undefined ? {} : { onProgress }) });
			worker.postMessage(request);
		});

	return {
		async init(modelUrl, onProgress) {
			await send({ id: nextId++, kind: "init", modelUrl }, onProgress);
		},
		async bestMove(board: number[], H: number, W: number, cfg: DepthCfg): Promise<AiMove> {
			const result = await send({ id: nextId++, kind: "move", board, H, W, cfg });
			return result ?? { dir: null as Dir | null, value: 0 };
		},
		dispose() {
			for (const entry of pending.values()) {
				entry.reject(new AiUnavailableError("NOT_READY", "the AI worker was disposed"));
			}
			pending.clear();
			worker.terminate();
		},
	};
}
