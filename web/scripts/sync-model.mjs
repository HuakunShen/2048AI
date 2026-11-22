/**
 * Copy the one committed model (`assets/model.pack`) into the web app's static
 * folder before dev and build.
 *
 * The pack is 33 MB and lives once, in `assets/`, so both hosts share it and it
 * is never duplicated in Git. `web/static/model/` stays gitignored.
 */
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../assets/model.pack");
const target = resolve(here, "../static/model/model.pack");

try {
	const existing = statSync(target);
	const wanted = statSync(source);
	if (existing.size === wanted.size && existing.mtimeMs >= wanted.mtimeMs) {
		process.exit(0);
	}
} catch {
	// Missing or unreadable: copy it.
}

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`[sync-model] ${source} -> ${target} (${String(statSync(target).size)} bytes)`);
