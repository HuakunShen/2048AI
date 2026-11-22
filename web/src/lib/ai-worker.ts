/// <reference lib="webworker" />
/**
 * The web app's AI Worker entry.
 *
 * Cloudflare Workers refuses a single static asset over 25 MiB and the pack is
 * 31.6 MiB, so the deployment ships it in parts and this loader concatenates
 * them before decoding. The bytes are identical to `assets/model.pack`; only
 * the number of requests differs. The Kunkun plugin page, served from its own
 * loopback origin, uses the single-file default instead.
 */
import {
	buildValueFromPacked,
	startAiWorker,
	type AiErrorCode,
	type UniversalValue,
	type WorkerScope
} from "@2048/core";

interface PartsManifest {
	bytes: number;
	count: number;
}

function failure(code: AiErrorCode, message: string): Error {
	return Object.assign(new Error(message), { code });
}

async function fetchParts(modelUrl: string, report: (fraction: number) => void): Promise<UniversalValue> {
	let manifestResponse: Response;
	try {
		manifestResponse = await fetch(modelUrl);
	} catch (cause) {
		throw failure("MODEL_MISSING", `parts manifest fetch failed: ${String(cause)}`);
	}
	if (!manifestResponse.ok) {
		throw failure("MODEL_MISSING", `parts manifest returned ${String(manifestResponse.status)}`);
	}
	const manifest = (await manifestResponse.json()) as PartsManifest;
	if (!Number.isSafeInteger(manifest.count) || manifest.count < 1 || !Number.isSafeInteger(manifest.bytes)) {
		throw failure("MODEL_CORRUPT", "parts manifest is invalid");
	}

	const base = new URL(modelUrl);
	const parts: Uint8Array[] = [];
	let loaded = 0;
	for (let i = 0; i < manifest.count; i += 1) {
		const partUrl = new URL(`model.pack.${String(i)}`, base).href;
		let response: Response;
		try {
			response = await fetch(partUrl);
		} catch (cause) {
			throw failure("MODEL_MISSING", `part ${String(i)} fetch failed: ${String(cause)}`);
		}
		if (!response.ok) {
			throw failure("MODEL_MISSING", `part ${String(i)} returned ${String(response.status)}`);
		}
		const bytes = new Uint8Array(await response.arrayBuffer());
		parts.push(bytes);
		loaded += bytes.byteLength;
		report(0.9 * Math.min(1, loaded / manifest.bytes));
	}

	if (loaded !== manifest.bytes) {
		throw failure("MODEL_CORRUPT", `model parts total ${String(loaded)} bytes, expected ${String(manifest.bytes)}`);
	}

	const joined = new Uint8Array(manifest.bytes);
	let offset = 0;
	for (const part of parts) {
		joined.set(part, offset);
		offset += part.byteLength;
	}

	try {
		const value = buildValueFromPacked(joined);
		report(1);
		return value;
	} catch (cause) {
		throw failure("MODEL_CORRUPT", cause instanceof Error ? cause.message : String(cause));
	}
}

startAiWorker(self as unknown as WorkerScope, { fetchModel: fetchParts });
