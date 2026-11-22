/**
 * Copy the committed model (`assets/model.pack`) into the web app's static
 * folder before dev and build — split into parts.
 *
 * Why split: Cloudflare Workers refuses a single static asset over 25 MiB, and
 * the pack is 31.6 MiB. Two parts of ≤ 20 MiB each keep the deployment legal;
 * the browser concatenates them before decoding, so the bytes are identical.
 * `web/static/model/` stays gitignored and is regenerated from `assets/`.
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const CHUNK_BYTES = 20 * 1024 * 1024;

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../assets/model.pack");
const targetDir = resolve(here, "../static/model");
const manifestPath = join(targetDir, "parts.json");

/** Rewrite the parts only when the pack or the split rule changed. */
function upToDate(total, count) {
	if (!existsSync(manifestPath)) return false;
	try {
		const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
		if (manifest.bytes !== total || manifest.count !== count) return false;
		for (let i = 0; i < count; i += 1) {
			if (!existsSync(join(targetDir, `model.pack.${String(i)}`))) return false;
			if (statSync(join(targetDir, `model.pack.${String(i)}`)).mtimeMs < statSync(source).mtimeMs) {
				return false;
			}
		}
		return true;
	} catch {
		return false;
	}
}

const bytes = readFileSync(source);
const count = Math.ceil(bytes.byteLength / CHUNK_BYTES);

if (upToDate(bytes.byteLength, count)) {
	process.exit(0);
}

mkdirSync(targetDir, { recursive: true });
for (let i = 0; i < count; i += 1) {
	const from = i * CHUNK_BYTES;
	const to = Math.min(bytes.byteLength, from + CHUNK_BYTES);
	const part = bytes.subarray(from, to);
	const file = join(targetDir, `model.pack.${String(i)}`);
	writeFileSync(file, part);
	console.log(`[sync-model] ${String(i)}: ${String(part.byteLength)} bytes -> ${file}`);
}
// The single-file copy is never produced here; Cloudflare would reject it.
// Remove one left by an older run so a stale asset cannot fail the build.
try {
	rmSync(join(targetDir, "model.pack"));
} catch {
	// Absent is the normal case.
}
writeFileSync(
	manifestPath,
	`${JSON.stringify({ bytes: bytes.byteLength, count, partBytes: CHUNK_BYTES }, null, 2)}\n`
);
console.log(
	`[sync-model] ${String(bytes.byteLength)} bytes in ${String(count)} parts (Cloudflare caps one asset at 25 MiB)`
);

// Keep the plugin's single-file path available for local poking, but never in static/.
if (process.env["TWENTY48_KEEP_SINGLE"] === "1") {
	copyFileSync(source, join(targetDir, "model.pack"));
}
