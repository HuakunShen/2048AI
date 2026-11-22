/**
 * Geometry probe: the cells must be square at any window shape, and nothing may
 * scroll. Serves a staged dist and measures the grid cells at several viewports
 * and board shapes.
 *
 *   node <this> <dist-dir>
 */
import { createServer } from "node:http";
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const dist = process.argv[2];
if (dist === undefined) throw new Error("usage: geometry-probe.mjs <dist-dir>");

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".pack": "application/octet-stream" };
const server = createServer((request, response) => {
	const url = new URL(request.url ?? "/", "http://127.0.0.1");
	const path = normalize(join(dist, url.pathname === "/" ? "index.html" : url.pathname));
	if (!path.startsWith(dist) || !existsSync(path)) {
		response.writeHead(404).end();
		return;
	}
	response.writeHead(200, { "content-type": MIME[extname(path)] ?? "application/octet-stream", "content-length": String(statSync(path).size) });
	createReadStream(path).pipe(response);
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const origin = `http://127.0.0.1:${String(server.address().port)}/`;

function launchOptions() {
	const cache = join(homedir(), "Library/Caches/ms-playwright");
	if (!existsSync(cache)) return {};
	const names = readdirSync(cache).filter((n) => n.startsWith("chromium-")).sort((a, b) => Number(b.split("-")[1]) - Number(a.split("-")[1]));
	for (const name of names) {
		const app = join(cache, name, "chrome-mac-arm64", "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing");
		if (existsSync(app)) return { executablePath: app };
	}
	return {};
}

const report = { origin, cases: [] };
const browser = await chromium.launch(await launchOptions());
try {
	for (const viewport of [{ width: 480, height: 720 }, { width: 900, height: 420 }, { width: 380, height: 900 }, { width: 1280, height: 700 }]) {
		const page = await browser.newPage({ viewport });
		await page.goto(origin, { waitUntil: "load" });
		await page.waitForFunction(() => document.querySelectorAll(".cell").length > 0, null, { timeout: 15000 });
		for (const shape of ["4×4", "4×5", "5×4", "6×6", "3×4"]) {
			await page.getByRole("button", { name: shape, exact: true }).click();
			await page.waitForTimeout(250);
			const measured = await page.evaluate(() => {
				const cells = Array.from(document.querySelectorAll(".cell")).slice(0, 6);
				const boxes = cells.map((c) => {
					const r = c.getBoundingClientRect();
					return { w: Math.round(r.width * 100) / 100, h: Math.round(r.height * 100) / 100 };
				});
				const board = document.querySelector(".board").getBoundingClientRect();
				const scroller = document.scrollingElement;
				return {
					boxes,
					board: { w: Math.round(board.width), h: Math.round(board.height) },
					scroll: { y: scroller.scrollHeight - scroller.clientHeight, x: scroller.scrollWidth - scroller.clientWidth },
				};
			});
			const worst = measured.boxes.reduce((acc, b) => Math.max(acc, Math.abs(b.w - b.h)), 0);
			report.cases.push({ viewport, shape, ...measured, maxCellDelta: Math.round(worst * 100) / 100 });
		}
		await page.close();
	}
} finally {
	await browser.close();
	server.close();
}
// Cells are square at every window shape. Scrolling is allowed only when the
// window is too short for the chrome plus the board's minimum (below 560 px);
// at the declared 480x720 nothing may scroll.
report.ok = report.cases.every(
	(c) => c.maxCellDelta <= 1 && c.scroll.x <= 0 && (c.viewport.height < 560 || c.scroll.y <= 0)
);
console.log(JSON.stringify(report, null, 2));
if (!report.ok) process.exitCode = 1;
