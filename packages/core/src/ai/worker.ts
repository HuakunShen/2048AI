/// <reference lib="webworker" />
/**
 * The AI Worker body.
 *
 * A host creates its own one-line worker entry —
 * `new Worker(new URL("./ai-worker.ts", import.meta.url), { type: "module" })` —
 * whose module calls {@link startAiWorker}. The host owns the bundling, so this
 * package never needs a framework- or bundler-specific worker import.
 *
 * One universal value function is shared across board sizes; one expectimax
 * searcher is compiled per shape on demand.
 */
import { Engine, type Board } from "../engine";
import { buildValueFromPacked } from "./model";
import { UniversalValue } from "./universal";
import { Expectimax } from "./expectimax";
import type { AiErrorCode, WorkerRequest, WorkerResponse } from "./protocol";

export interface WorkerScope {
	onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
	postMessage(message: WorkerResponse): void;
}

/**
 * How the worker gets its tables.
 *
 * The default fetches one file and decodes it. A host that must split the model
 * (Cloudflare Workers caps a single asset at 25 MiB, and the pack is 31.6 MiB)
 * supplies its own: it receives the `modelUrl` from `init` and reports progress.
 */
export interface WorkerOptions {
	fetchModel?: (modelUrl: string, report: (fraction: number) => void) => Promise<UniversalValue>;
}

export function startAiWorker(scope: WorkerScope, options: WorkerOptions = {}): void {
	let value: UniversalValue | null = null;
	let loading: Promise<UniversalValue> | null = null;
	const searchers = new Map<string, Expectimax>();

	const searcherFor = (H: number, W: number): Expectimax => {
		const key = `${H}x${W}`;
		let searcher = searchers.get(key);
		if (searcher === undefined) {
			const engine = new Engine(H, W);
			searcher = new Expectimax(engine, (board: Board) => value!.value(board, H, W));
			searchers.set(key, searcher);
		}
		return searcher;
	};

	const fetchSingle = async (
		modelUrl: string,
		report: (fraction: number) => void
	): Promise<UniversalValue> => {
		let response: Response;
		try {
			response = await fetch(modelUrl);
		} catch (cause) {
			throw Object.assign(new Error(`model fetch failed: ${String(cause)}`), {
				code: "MODEL_MISSING" satisfies AiErrorCode,
			});
		}
		if (!response.ok) {
			throw Object.assign(new Error(`model fetch returned ${String(response.status)}`), {
				code: "MODEL_MISSING" satisfies AiErrorCode,
			});
		}
		const bytes = new Uint8Array(await response.arrayBuffer());
		report(0.5);
		try {
			const built = buildValueFromPacked(bytes);
			report(1);
			return built;
		} catch (cause) {
			throw Object.assign(new Error(cause instanceof Error ? cause.message : String(cause)), {
				code: "MODEL_CORRUPT" satisfies AiErrorCode,
			});
		}
	};

	const fetchModel = options.fetchModel ?? fetchSingle;

	const codeOf = (error: unknown): AiErrorCode => {
		if (typeof error === "object" && error !== null && "code" in error) {
			const code = (error as { code?: unknown }).code;
			if (code === "MODEL_MISSING" || code === "MODEL_CORRUPT") return code;
		}
		return "MODEL_CORRUPT";
	};

	scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
		const request = event.data;
		void (async () => {
			if (request.kind === "init") {
				if (value !== null) {
					scope.postMessage({ id: request.id, kind: "ready" });
					return;
				}
				try {
					loading ??= fetchModel(request.modelUrl, (fraction) =>
						scope.postMessage({ id: request.id, kind: "progress", fraction })
					);
					value = await loading;
					scope.postMessage({ id: request.id, kind: "ready" });
				} catch (error) {
					loading = null;
					scope.postMessage({
						id: request.id,
						kind: "error",
						code: codeOf(error),
						message: error instanceof Error ? error.message : String(error),
					});
				}
				return;
			}

			if (value === null) {
				scope.postMessage({
					id: request.id,
					kind: "error",
					code: "NOT_READY",
					message: "the AI model is not loaded",
				});
				return;
			}

			try {
				const { dir, value: score } = searcherFor(request.H, request.W).getMove(
					Uint8Array.from(request.board) as Board,
					request.cfg
				);
				scope.postMessage({ id: request.id, kind: "move", dir, value: score });
			} catch (cause) {
				// A search that throws must answer, or the caller waits forever.
				scope.postMessage({
					id: request.id,
					kind: "error",
					code: "NOT_READY",
					message: `search failed: ${cause instanceof Error ? cause.message : String(cause)}`,
				});
			}
		})();
	};
}
