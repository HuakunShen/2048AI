import { readFileSync } from "node:fs";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildValue, buildValueFromPacked, type Manifest } from "../vendor/model";
import { UniversalValue } from "../vendor/universal";

/**
 * Load the trained n-tuple weights from disk.
 *
 * The release ships one compact ~32 MB `model.pack` file. The older web-style
 * sharded export remains supported for developers that set an external model
 * directory, and the game still degrades to human play when no model exists.
 *
 * The tables are stored sparsely and sharded (`lut{k}_{p}.bin`, each a
 * self-contained `uint32 idx ++ float32 val` sub-list) because a single table
 * exceeds the 25 MiB asset limit upstream. `buildValue` scatters the parts back
 * into dense Float32Arrays. The packed form uses delta-varint indices and
 * int16 values, reducing the distribution from roughly 84 MB to one file.
 */
export const DEFAULT_MODEL_DIR = fileURLToPath(new URL("../../model/", import.meta.url));
const PACKED_MODEL = "model.pack";

/** Where the weights live: `$UNIVIEW_2048_MODEL_DIR`, else `<package>/model/`. */
export function modelDir(): string {
  return process.env.UNIVIEW_2048_MODEL_DIR ?? DEFAULT_MODEL_DIR;
}

/** True when a usable model is present at {@link modelDir}. */
export function hasModel(dir: string = modelDir()): boolean {
  return existsSync(join(dir, PACKED_MODEL)) || existsSync(join(dir, "manifest.json"));
}

/**
 * Read the manifest + every LUT shard and rebuild the value function.
 * Returns null when no model is installed. Throws only if the model is present
 * but corrupt — a missing shard is a real error, not a reason to play dumb.
 */
export function loadModel(dir: string = modelDir()): UniversalValue | null {
  if (!hasModel(dir)) return null;

  const packed = join(dir, PACKED_MODEL);
  if (existsSync(packed)) return buildValueFromPacked(readFileSync(packed));

  const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")) as Manifest;
  const buffers = manifest.patterns.map((_pattern, k) =>
    manifest.parts[k]!.map((_nnz, p) => {
      const raw = readFileSync(join(dir, `lut${k}_${p}.bin`));
      // Slice to a standalone ArrayBuffer: Node pools small Buffers, so
      // `raw.buffer` may be a shared arena much larger than this file.
      return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    }),
  );
  return buildValue(manifest, buffers);
}
