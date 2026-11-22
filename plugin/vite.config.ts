import { copyFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

const modelSource = fileURLToPath(new URL("../assets/model.pack", import.meta.url));

/**
 * Copy the one committed model next to the page, so `fetch("model.pack")`
 * resolves relative to `index.html` wherever a host serves it. `publicDir` is
 * off: `assets/` also holds a test fixture that has no business in the page.
 */
function copyModel(): Plugin {
	return {
		name: "copy-model",
		apply: "build",
		closeBundle() {
			const target = fileURLToPath(new URL("dist/model.pack", import.meta.url));
			copyFileSync(modelSource, target);
			this.info(`copied model.pack (${String(statSync(target).size)} bytes)`);
		},
	};
}

export default defineConfig({
	// Relative URLs only: Kunkun serves this page from a per-extension loopback
	// origin at a path the bundle cannot know at build time.
	base: "./",
	plugins: [svelte(), copyModel()],
	publicDir: false,
	build: {
		outDir: "dist",
		emptyOutDir: true,
	},
});
