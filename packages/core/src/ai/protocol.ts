/**
 * The message protocol between the page and the AI Worker.
 *
 * Deliberately tiny and dependency-free: the model is 33 MB and the search runs
 * off the UI thread, and nothing else is needed to talk to it. Both sides import
 * these types, so a shape change is a compile error rather than a silent hang.
 */
import type { Dir } from "../engine";
import type { DepthCfg } from "./expectimax";

export type AiErrorCode = "MODEL_MISSING" | "MODEL_CORRUPT" | "NOT_READY" | "TIMEOUT";

export type WorkerRequest =
	| { id: number; kind: "init"; modelUrl: string }
	| { id: number; kind: "move"; board: number[]; H: number; W: number; cfg: DepthCfg };

export type WorkerResponse =
	| { id: number; kind: "progress"; fraction: number }
	| { id: number; kind: "ready" }
	| { id: number; kind: "error"; code: AiErrorCode; message: string }
	| { id: number; kind: "move"; dir: Dir | null; value: number };
