/**
 * Browser smoke test for the frame page Kunkun will load.
 *
 * Serves `plugin/dist` on loopback and drives it with Chromium: the custom
 * element upgrades, the board renders tiles, arrow keys move, the AI model
 * finishes loading from the served `model.pack`, and auto-play advances the
 * game. Prints one JSON object; exits 1 on any failed check.
 *
 * Run from the repository root: `node web/scripts/smoke-plugin.mjs`
 * (`playwright` resolves from `web/node_modules`).
 */
import { createServer } from "node:http";
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

/**
 * Use a Chromium that is actually installed. Playwright pins a build number that
 * may not be downloaded here; any newer cached one drives this page fine.
 */
async function launchOptions() {
	if (process.env["PLAYWRIGHT_CHROMIUM"] !== undefined) {
		return { executablePath: process.env["PLAYWRIGHT_CHROMIUM"] };
	}
	const cache = join(homedir(), "Library/Caches/ms-playwright");
	if (!existsSync(cache)) return {};
	const candidates = readdirSync(cache)
		.filter((name) => name.startsWith("chromium-"))
		.sort((a, b) => Number(b.split("-")[1]) - Number(a.split("-")[1]));
	for (const name of candidates) {
		for (const flavour of ["chrome-mac-arm64", "chrome-mac", "chrome-linux"]) {
			const app = join(cache, name, flavour, "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing");
			if (existsSync(app)) return { executablePath: app };
			const plain = join(cache, name, flavour, "chrome");
			if (existsSync(plain)) return { executablePath: plain };
		}
	}
	return {};
}

const dist = process.env["SMOKE_DIST"] ?? fileURLToPath(new URL("../../plugin/dist/", import.meta.url));
const served = [];

const MIME = {
	".html": "text/html; charset=utf-8",
	".js": "text/javascript; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".json": "application/json; charset=utf-8",
	".pack": "application/octet-stream",
	".svg": "image/svg+xml",
};

const server = createServer((request, response) => {
	const url = new URL(request.url ?? "/", "http://127.0.0.1");
	const path = normalize(join(dist, url.pathname === "/" ? "index.html" : url.pathname));
	if (!path.startsWith(dist)) {
		response.writeHead(403).end();
		return;
	}
	try {
		const info = statSync(path);
		if (!info.isFile()) throw new Error("not a file");
		response.writeHead(200, {
			"content-type": MIME[extname(path)] ?? "application/octet-stream",
			"content-length": String(info.size),
		});
		createReadStream(path).pipe(response);
	} catch {
		served.push(`404 ${url.pathname}`);
		response.writeHead(404).end("not found");
	}
});

await new Promise((done) => server.listen(0, "127.0.0.1", done));
const port = server.address().port;
const origin = `http://127.0.0.1:${String(port)}/`;

const results = { origin, checks: [] };
function check(name, ok, detail) {
	results.checks.push({ name, ok, detail });
	if (!ok) process.exitCode = 1;
}

const browser = await chromium.launch(await launchOptions());
try {
	const page = await browser.newPage();
	const consoleErrors = [];
	page.on("console", (message) => {
		if (message.type() === "error") consoleErrors.push(message.text());
	});
	page.on("pageerror", (error) => consoleErrors.push(String(error)));
	const badResponses = [];
	page.on("response", (response) => {
		if (response.status() >= 400) badResponses.push(`${String(response.status())} ${response.url()}`);
	});

	await page.goto(origin, { waitUntil: "load" });

	check(
		"custom element defined",
		await page.evaluate(() => customElements.get("game-2048") !== undefined)
	);

	await page.waitForFunction(() => document.querySelectorAll(".t2048-inner").length >= 2, null, {
		timeout: 15_000,
	});
	const tiles = await page.evaluate(() => document.querySelectorAll(".t2048-inner").length);
	check("board renders tiles", tiles >= 2, { tiles });

	const movesBefore = await page.evaluate(() => document.body.innerText.match(/(\d+) moves/)?.[1] ?? "0");
	await page.keyboard.press("ArrowLeft");
	await page.waitForTimeout(300);
	const movesAfter = await page.evaluate(() => document.body.innerText.match(/(\d+) moves/)?.[1] ?? "0");
	check("arrow key moves the board", Number(movesAfter) > Number(movesBefore), {
		movesBefore,
		movesAfter,
	});

	// The model is 33 MB over loopback; give the decode room.
	let modelLoaded = true;
	try {
		await page.waitForFunction(
			() => !document.body.innerText.includes("Loading AI model"),
			null,
			{ timeout: 60_000 }
		);
	} catch {
		modelLoaded = false;
	}
	if (!modelLoaded) {
		const diagnostic = await page.evaluate(() => document.body.innerText.slice(0, 600));
		check("AI model loaded", false, { diagnostic, consoleErrors: consoleErrors.slice(0, 5), badResponses });
	} else {
		const ai = await page.evaluate(() => {
			const buttons = Array.from(document.querySelectorAll("button"));
			const auto = buttons.find((button) => /Auto-play|Pause/.test(button.textContent ?? ""));
			return {
				autoDisabled: auto?.hasAttribute("disabled") ?? true,
				text: document.body.innerText.slice(0, 400),
			};
		});
		check("AI model loaded and auto-play enabled", ai.autoDisabled === false, ai);
	}

	if (modelLoaded) {
		const beforeAuto = await page.evaluate(() => document.body.innerText.match(/(\d+) moves/)?.[1] ?? "0");
		await page.getByRole("button", { name: "Auto-play" }).click();
		await page.waitForTimeout(4000);
		const afterAuto = await page.evaluate(() => document.body.innerText.match(/(\d+) moves/)?.[1] ?? "0");
		check("auto-play advances the game", Number(afterAuto) > Number(beforeAuto), {
			beforeAuto,
			afterAuto,
		});
	}

	check("no console errors", consoleErrors.length === 0, { consoleErrors: consoleErrors.slice(0, 5), badResponses, served });
	results.ok = results.checks.every((entry) => entry.ok);
} finally {
	await browser.close();
	server.close();
}

console.log(JSON.stringify(results, null, 2));
