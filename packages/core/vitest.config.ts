import { defineConfig } from "vitest/config";

// A config file in the package stops Vitest from walking up to the Kunkun
// repository's own vitest.config.ts, which lives above this submodule.
export default defineConfig({
	test: {
		environment: "node",
		include: ["test/**/*.test.ts"],
		testTimeout: 130_000,
	},
});
