import { describe, expect, it } from "vitest";
import { buildValueFromPacked, readPackedManifest } from "../src/ai/model";
import { AiUnavailableError, createWorkerAi } from "../src/ai/client";
import type { WorkerRequest, WorkerResponse } from "../src/ai/protocol";
import { packBytes } from "./load-model";

/**
 * The decoder is the one place where corrupt bytes could silently weaken the AI,
 * and the client is where a dead Worker could hang the UI. Both are pure enough
 * to test without a browser.
 */
describe("packed model decoder", () => {
	it("reads the header of the committed pack", () => {
		const { manifest, dataOffset } = readPackedManifest(packBytes());
		expect(manifest.alphabet).toBe(16);
		expect(manifest.tables).toHaveLength(5);
		expect(manifest.patterns.map((p) => p.id)).toEqual([
			"square_2x2",
			"line_4",
			"l_4",
			"rect_2x3",
			"corner_6",
		]);
		expect(dataOffset).toBeGreaterThan(12);
	});

	it("rejects a wrong magic, a truncated body and trailing bytes", () => {
		const bytes = packBytes();
		const wrongMagic = bytes.slice();
		wrongMagic[0] = 0x58; // 'X'
		expect(() => buildValueFromPacked(wrongMagic)).toThrow(/magic/);

		expect(() => buildValueFromPacked(bytes.slice(0, 64))).toThrow(/truncated/);

		const trailing = new Uint8Array(bytes.byteLength + 1);
		trailing.set(bytes);
		expect(() => buildValueFromPacked(trailing)).toThrow(/trailing/);
	});

	it("rejects a header that claims more tables than it has", () => {
		const bytes = packBytes();
		const { manifest, dataOffset } = readPackedManifest(bytes);
		const broken = JSON.parse(JSON.stringify(manifest)) as typeof manifest;
		broken.patterns = broken.patterns.slice(0, 4);
		const json = new TextEncoder().encode(JSON.stringify(broken));
		const header = new Uint8Array(12 + json.byteLength);
		header.set(bytes.subarray(0, 8)); // magic
		new DataView(header.buffer).setUint32(8, json.byteLength, true);
		header.set(json, 12);
		const combined = new Uint8Array(header.byteLength + (bytes.byteLength - dataOffset));
		combined.set(header);
		combined.set(bytes.subarray(dataOffset), header.byteLength);
		expect(() => buildValueFromPacked(combined)).toThrow(/mismatch|invalid/);
	});
});

/** A Worker stand-in: records what the client posted and lets a test answer. */
function fakeWorker() {
	const listeners = new Map<string, ((event: unknown) => void)[]>();
	const posted: WorkerRequest[] = [];
	const worker = {
		posted,
		addEventListener(type: string, listener: (event: unknown) => void) {
			const list = listeners.get(type) ?? [];
			list.push(listener);
			listeners.set(type, list);
		},
		postMessage(message: WorkerRequest) {
			posted.push(message);
		},
		terminate() {},
		emit(type: string, event: unknown) {
			for (const listener of listeners.get(type) ?? []) listener(event);
		},
		answer(message: WorkerResponse) {
			for (const listener of listeners.get("message") ?? []) listener({ data: message });
		},
	};
	return worker;
}

describe("AI client", () => {
	it("rejects a pending request when the worker errors", async () => {
		const worker = fakeWorker();
		const ai = createWorkerAi(worker as unknown as Worker);
		const pending = ai.bestMove([0], 4, 4, { depth: 1 });
		worker.emit("error", { message: "chunk failed to load" });
		await expect(pending).rejects.toBeInstanceOf(AiUnavailableError);
		await expect(pending).rejects.toMatchObject({ code: "NOT_READY" });
	});

	it("times out a request the worker never answers", async () => {
		const worker = fakeWorker();
		const ai = createWorkerAi(worker as unknown as Worker, { moveTimeoutMs: 10 });
		await expect(ai.bestMove([0], 4, 4, { depth: 1 })).rejects.toMatchObject({ code: "TIMEOUT" });
		expect(worker.posted).toHaveLength(1);
		expect(worker.posted[0]?.kind).toBe("move");
	});

	it("answers a move and a progress report from the worker", async () => {
		const worker = fakeWorker();
		const ai = createWorkerAi(worker as unknown as Worker);
		const progress: number[] = [];
		const init = ai.init("model.pack", (fraction) => progress.push(fraction));
		const initRequest = worker.posted.at(-1);
		expect(initRequest?.kind).toBe("init");
		worker.answer({ id: initRequest!.id, kind: "progress", fraction: 0.5 });
		worker.answer({ id: initRequest!.id, kind: "ready" });
		await expect(init).resolves.toBeUndefined();
		expect(progress).toEqual([0.5]);

		const move = ai.bestMove([0], 4, 4, { depth: 1 });
		const moveRequest = worker.posted.at(-1);
		worker.answer({ id: moveRequest!.id, kind: "move", dir: "LEFT", value: 12 });
		await expect(move).resolves.toEqual({ dir: "LEFT", value: 12 });
	});
});
