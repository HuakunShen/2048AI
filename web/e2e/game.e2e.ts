import { expect, test } from '@playwright/test';

/**
 * Full-stack smoke test: loads the real app, waits for the Web Worker to download
 * and scatter the weights (via kkrpc `init`), starts auto-play, and confirms the
 * browser-side AI actually advances the game to a high tile. Exercises the worker
 * + kkrpc + engine + UI together.
 */
test('boots the in-browser AI and auto-plays to a high tile', async ({ page }) => {
	const pageErrors: string[] = [];
	page.on('pageerror', (e) => pageErrors.push(String(e)));

	await page.goto('/');

	// The board renders immediately (client-side).
	await expect(page.getByRole('heading', { name: /2048/ })).toBeVisible();

	// Auto-play stays disabled until the worker finishes loading the ~22MB model.
	const auto = page.getByRole('button', { name: /Auto-play/i });
	await expect(auto).toBeEnabled({ timeout: 90_000 });

	await auto.click();

	// The Max-tile badge should climb well past the initial 2/4 as the AI plays.
	await expect
		.poll(
			async () => {
				const txt = (await page.getByText(/^Max/).first().textContent()) ?? '';
				return parseInt(txt.replace(/[^0-9]/g, ''), 10) || 0;
			},
			{ timeout: 90_000, intervals: [500, 1000, 2000] }
		)
		.toBeGreaterThanOrEqual(256);

	expect(pageErrors, `uncaught page errors:\n${pageErrors.join('\n')}`).toEqual([]);
});
