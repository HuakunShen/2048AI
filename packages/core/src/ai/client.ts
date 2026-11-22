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

export interface WorkerAiOptions {
	/** Budget for the model fetch and decode. Default 180 s. */
	readonly initTimeoutMs?: number;
	/** Budget for one search. Default 60 s. */
	readonly moveTimeoutMs?: number;
}

export function createWorkerAi(
	worker: Worker,
	options: WorkerAiOptions = {}
): AiProvider & { dispose(): void } {
	const initTimeoutMs = options.initTimeoutMs ?? 180_000;
	const moveTimeoutMs = options.moveTimeoutMs ?? 60_000;
	let nextId = 1;
	const pending = new Map<number, Pending>();

	/** Reject everything in flight: a dead worker must not leave a promise hanging. */
	const failAll = (code: AiErrorCode, message: string): void => {
		for (const entry of pending.values()) entry.reject(new AiUnavailableError(code, message));
		pending.clear();
	};

	worker.addEventListener("error", (event) => {
		failAll("NOT_READY", `the AI worker failed: ${String((event as ErrorEvent).message ?? "unknown error")}`);
	});
	worker.addEventListener("messageerror", () => {
		failAll("NOT_READY", "the AI worker sent a message that could not be decoded");
	});

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

	const send = (
		request: WorkerRequest,
		timeoutMs: number,
		onProgress?: (fraction: number) => void
	): Promise<AiMove | void> =>
		new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				if (!pending.delete(request.id)) return;
				reject(new AiUnavailableError("TIMEOUT", `the AI worker did not answer within ${String(timeoutMs)} ms`));
			}, timeoutMs);
			const settle = {
				resolve: (value: AiMove | void) => {
					clearTimeout(timer);
					resolve(value);
				},
				reject: (reason: unknown) => {
					clearTimeout(timer);
					reject(reason);
				},
				...(onProgress === undefined ? {} : { onProgress }),
			};
			pending.set(request.id, settle);
			worker.postMessage(request);
		});

	return {
		async init(modelUrl, onProgress) {
			// The 33 MB fetch and the decode of ~33.7 M entries happen inside this budget.
			await send({ id: nextId++, kind: "init", modelUrl }, initTimeoutMs, onProgress);
		},
		async bestMove(board: number[], H: number, W: number, cfg: DepthCfg): Promise<AiMove> {
			const result = await send({ id: nextId++, kind: "move", board, H, W, cfg }, moveTimeoutMs);
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
