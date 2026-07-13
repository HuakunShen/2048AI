import { defineConfig } from '@playwright/test';

export default defineConfig({
	// The app is fully client-side, so a dev server is enough to exercise the real
	// worker + UI + AI. (The scaffold's build+wrangler preview is heavier and its
	// port doesn't match; dev on a fixed port is the reliable e2e target.)
	webServer: {
		command: 'vite dev --port 4173',
		port: 4173,
		reuseExistingServer: true,
		timeout: 120_000
	},
	use: { baseURL: 'http://localhost:4173', headless: true },
	testMatch: '**/*.e2e.{ts,js}',
	timeout: 120_000
});
