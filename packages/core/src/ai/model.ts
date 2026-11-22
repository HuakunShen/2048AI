/**
 * Load the trained Universal model into a {@link UniversalValue}.
 *
 * The only supported format is the compact `2048NTP1` pack (`assets/model.pack`):
 * delta-varint indices followed by signed int16 values, scaled per table.
 * The older sharded `manifest.json` + `lut*_*.bin` export is gone — no such
 * files exist in the repository, and two of the web tests used to read them
 * from a gitignored, empty folder.
 *
 * This module is pure: no `fetch`, no filesystem. Callers pass bytes.
 */
import { UniversalValue } from "./universal";
import type { Pattern } from "./patterns";

export interface PackedTable {
	size: number;
	nnz: number;
	scale: number;
	idxBytes: number;
}

export interface PackedManifest {
	alphabet: number;
	patterns: { id: string; cells: [number, number][] }[];
	tables: PackedTable[];
}

export function patternsFromManifest(m: PackedManifest): Pattern[] {
	return m.patterns.map((p) => ({ id: p.id, cells: p.cells, alphabet: m.alphabet }));
}

const PACK_MAGIC = "2048NTP1";
const textDecoder = new TextDecoder();

/**
 * Read the pack header without decoding any table.
 *
 * Exported so a test can derive the quantisation bound from the real table
 * scales instead of hard-coding a tolerance, and so a host can show the model's
 * dimensions before paying the decode cost.
 */
export function readPackedManifest(bytes: Uint8Array): { manifest: PackedManifest; dataOffset: number } {
	if (bytes.byteLength < 12) throw new Error("packed model is truncated");
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const magic = textDecoder.decode(bytes.subarray(0, 8));
	if (magic !== PACK_MAGIC) throw new Error(`unsupported packed model magic: ${magic}`);
	const jsonLength = view.getUint32(8, true);
	const jsonStart = 12;
	const jsonEnd = jsonStart + jsonLength;
	if (jsonEnd > bytes.byteLength) throw new Error("packed model manifest is truncated");
	const manifest = JSON.parse(textDecoder.decode(bytes.subarray(jsonStart, jsonEnd))) as PackedManifest;
	if (!Array.isArray(manifest.patterns) || !Array.isArray(manifest.tables)) {
		throw new Error("packed model manifest is invalid");
	}
	if (manifest.patterns.length !== manifest.tables.length) {
		throw new Error("packed model pattern/table count mismatch");
	}
	return { manifest, dataOffset: jsonEnd };
}

/**
 * Decode the compact model format.
 *
 * Each table is stored as delta-varint indices followed by signed int16 values;
 * multiplying by the table scale recovers the original float32 sparse export.
 * The decoder validates offsets and indices so a partial/corrupt release fails
 * during startup instead of silently weakening the AI.
 */
export function buildValueFromPacked(bytes: Uint8Array): UniversalValue {
	const { manifest, dataOffset } = readPackedManifest(bytes);
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const luts: Float32Array[] = [];
	let offset = dataOffset;

	for (const table of manifest.tables) {
		if (!Number.isSafeInteger(table.size) || table.size <= 0 ||
			!Number.isSafeInteger(table.nnz) || table.nnz < 0 ||
			!Number.isSafeInteger(table.idxBytes) || table.idxBytes < 0 ||
			!Number.isFinite(table.scale)) {
			throw new Error("packed model table metadata is invalid");
		}
		const indexEnd = offset + table.idxBytes;
		const valueEnd = indexEnd + table.nnz * 2;
		if (valueEnd > bytes.byteLength) throw new Error("packed model table is truncated");

		const lut = new Float32Array(table.size);
		let indexOffset = offset;
		let index = 0;
		for (let i = 0; i < table.nnz; i += 1) {
			let delta = 0;
			let shift = 0;
			for (;;) {
				if (indexOffset >= indexEnd || shift > 28) {
					throw new Error("packed model index stream is invalid");
				}
				const byte = bytes[indexOffset++];
				delta += (byte & 0x7f) * 2 ** shift;
				if ((byte & 0x80) === 0) break;
				shift += 7;
			}
			index += delta;
			if (index < 0 || index >= table.size) throw new Error("packed model index is out of range");
			lut[index] = view.getInt16(indexEnd + i * 2, true) * table.scale;
		}
		if (indexOffset !== indexEnd) throw new Error("packed model index stream has trailing bytes");
		luts.push(lut);
		offset = valueEnd;
	}

	if (offset !== bytes.byteLength) throw new Error("packed model has trailing bytes");
	return new UniversalValue(patternsFromManifest(manifest), luts);
}
