import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildValueFromPacked, readPackedManifest, type PackedManifest } from "../src/ai/model";
import type { UniversalValue } from "../src/ai/universal";

/**
 * The one artefact both hosts share: `assets/model.pack`, the trained universal
 * n-tuple tables in the compact `2048NTP1` form (delta-varint indices +
 * int16 values). 33,110,705 bytes.
 */
const PACK = fileURLToPath(new URL("../../../assets/model.pack", import.meta.url));

export function packBytes(): Uint8Array {
	return new Uint8Array(readFileSync(PACK));
}

export function loadValue(): UniversalValue {
	return buildValueFromPacked(packBytes());
}

export function packManifest(): PackedManifest {
	return readPackedManifest(packBytes()).manifest;
}

/**
 * The worst error the packed form can add to V(), derived from the real scales.
 *
 * Each table stores `int16 × scale`, so one table lookup is off by at most
 * `scale / 2`. V() averages one table's lookups over its placements and sums the
 * five patterns, so an average of errors each within ±scale/2 is still within
 * ±scale/2, and the total is bounded by the sum over tables.
 */
export function quantisationBound(): number {
	return packManifest().tables.reduce((total, table) => total + table.scale / 2, 0);
}
