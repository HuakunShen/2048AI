/**
 * Load the exported Universal model into a {@link UniversalValue}.
 *
 * The web export is sparse and chunked (`uint32 idx ++ float32 val` parts),
 * while the terminal package uses the smaller `2048NTP1` packed form. Both
 * formats end up as the same dense lookup tables for fast expectimax reads.
 */
import { UniversalValue } from './universal';
import type { Pattern } from './patterns';

export interface Manifest {
	alphabet: number;
	patterns: { id: string; cells: [number, number][] }[];
	tableSizes: number[];
	parts: number[][]; // parts[k] = [nnz per part] for table k
	shapes: string[];
}

export interface PackedManifest {
	alphabet: number;
	patterns: { id: string; cells: [number, number][] }[];
	tables: { size: number; nnz: number; scale: number; idxBytes: number }[];
}

export function patternsFromManifest(m: Manifest): Pattern[] {
	return m.patterns.map((p) => ({ id: p.id, cells: p.cells, alphabet: m.alphabet }));
}

/** Scatter one part buffer (`uint32 idx ++ f32 val`) into a dense table. */
export function scatterPart(lut: Float32Array, buf: ArrayBuffer, nnz: number): void {
	const indices = new Uint32Array(buf, 0, nnz);
	const values = new Float32Array(buf, nnz * 4, nnz);
	for (let i = 0; i < nnz; i++) lut[indices[i]] = values[i];
}

/** Reconstruct one dense table from its part buffers. */
export function reconstructTable(size: number, partBufs: ArrayBuffer[], partNnz: number[]): Float32Array {
	const lut = new Float32Array(size);
	for (let p = 0; p < partBufs.length; p++) scatterPart(lut, partBufs[p], partNnz[p]);
	return lut;
}

/** Build a UniversalValue from a manifest + per-pattern part buffers. */
export function buildValue(manifest: Manifest, buffers: ArrayBuffer[][]): UniversalValue {
	const luts = manifest.tableSizes.map((size, k) =>
		reconstructTable(size, buffers[k], manifest.parts[k])
	);
	return new UniversalValue(patternsFromManifest(manifest), luts);
}

const PACK_MAGIC = "2048NTP1";
const textDecoder = new TextDecoder();

function readPackedManifest(bytes: Uint8Array): { manifest: PackedManifest; dataOffset: number } {
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
 * Decode the compact terminal model format.
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
	return new UniversalValue(
		manifest.patterns.map((p) => ({ id: p.id, cells: p.cells, alphabet: manifest.alphabet })),
		luts,
	);
}
