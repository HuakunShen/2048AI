import { describe, expect, it } from "vitest";
import { Engine, DIRS, tilesToBoard, type Board, type Dir } from "../src/engine";
import { finalizeSprites, slideSprites, spritesFromBoard, type Sprite } from "../src/game/sprites";
import { GameController, type Scheduler } from "../src/game/controller";
import { BEST_KEY, memoryStorage, type GameStorage } from "../src/game/storage";

/** Deterministic RNG (mulberry32), so a game is reproducible. */
function seeded(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** A scheduler the test drives by hand, so the two phases are observable. */
function manualScheduler(): Scheduler & { flush(): void; pending(): number } {
	let queue: Array<() => void> = [];
	return {
		after(_ms, fn) {
			queue.push(fn);
			return () => {
				queue = queue.filter((entry) => entry !== fn);
			};
		},
		flush() {
			const run = queue;
			queue = [];
			for (const fn of run) fn();
		},
		pending() {
			return queue.length;
		},
	};
}

function fromTiles(rows: number[][]): Board {
	return tilesToBoard(rows.flat());
}

function firstChangingDir(engine: Engine, board: Board): Dir {
	for (const dir of DIRS) if (engine.move(board, dir).changed) return dir;
	throw new Error("no changing direction");
}

describe("sprite model", () => {
	it("collapses a merged pair and keeps one sprite per cell", () => {
		const engine = new Engine(4, 4);
		const board = fromTiles([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
		const plan = engine.planMove(board, "LEFT");
		let id = 0;
		let pop = 0;
		const sprites: Sprite[] = spritesFromBoard(board, () => id++, () => ++pop);

		const merged = slideSprites(sprites, plan.slides);
		expect(merged.has(0)).toBe(true);
		// Both halves are still present and now share a cell, before the finalise.
		expect(new Set(sprites.map((s) => s.index)).size).toBe(1);

		const final = finalizeSprites({
			sprites,
			board: plan.after,
			mergedCells: merged,
			spawned: -1,
			allocId: () => id++,
			allocPop: () => ++pop,
		});
		expect(final).toHaveLength(1);
		expect(final[0]!.index).toBe(0);
		expect(final[0]!.exp).toBe(2); // two 2s merged into a 4
		expect(final[0]!.popKind).toBe("merge");
	});

	it("accounts for every source cell exactly once", () => {
		const engine = new Engine(4, 4);
		const board = fromTiles([[2, 0, 4, 0], [0, 8, 0, 8], [16, 0, 0, 0], [0, 0, 0, 2]]);
		const plan = engine.planMove(board, "RIGHT");
		let id = 0;
		let pop = 0;
		const sprites = spritesFromBoard(board, () => id++, () => ++pop);
		const sources = sprites.map((s) => s.index).sort((a, b) => a - b);
		slideSprites(sprites, plan.slides);
		expect(plan.slides.map((s) => s.from).sort((a, b) => a - b)).toEqual(sources);
	});
});

describe("controller", () => {
	it("commits the slide first and the spawn after the slide delay", () => {
		const scheduler = manualScheduler();
		const controller = new GameController({ rng: seeded(7), scheduler });
		const before = controller.getSnapshot();
		const nonEmptyBefore = Array.from(before.board).filter((v) => v !== 0).length;
		const dir = firstChangingDir(controller.engine, before.board);

		expect(controller.humanMove(dir)).toBe(true);
		const sliding = controller.getSnapshot();
		// Slides preserve the sprite count; the spawned tile has no sprite yet,
		// even though the board already contains it.
		expect(sliding.sprites).toHaveLength(nonEmptyBefore);
		expect(Array.from(sliding.board).filter((v) => v !== 0).length).toBeGreaterThanOrEqual(nonEmptyBefore);
		expect(scheduler.pending()).toBe(1);

		scheduler.flush();
		const settled = controller.getSnapshot();
		expect(settled.sprites).toHaveLength(Array.from(settled.board).filter((v) => v !== 0).length);
		expect(new Set(settled.sprites.map((s) => s.index)).size).toBe(settled.sprites.length);
		expect(settled.moves).toBe(1);
	});

	it("is reproducible from a seed", () => {
		const play = (seed: number): number[] => {
			const scheduler = manualScheduler();
			const controller = new GameController({ rng: seeded(seed), scheduler });
			const boards: number[] = [];
			for (const dir of ["LEFT", "DOWN", "RIGHT", "UP"] as const) {
				controller.humanMove(dir);
				scheduler.flush();
			}
			boards.push(...Array.from(controller.getSnapshot().board));
			boards.push(controller.getSnapshot().score);
			return boards;
		};
		expect(play(42)).toEqual(play(42));
		expect(play(42)).not.toEqual(play(43));
	});

	it("persists the best score through the injected storage", () => {
		const storage: GameStorage = memoryStorage();
		const scheduler = manualScheduler();
		const controller = new GameController({ rng: seeded(11), scheduler, storage });
		const dir = firstChangingDir(controller.engine, controller.getSnapshot().board);
		controller.humanMove(dir);
		scheduler.flush();
		const { score, best } = controller.getSnapshot();
		expect(best).toBeGreaterThanOrEqual(score);
		if (score > 0) expect(storage.get(BEST_KEY)).toBe(String(best));
	});

	it("keeps human play working when the AI never loads", async () => {
		const scheduler = manualScheduler();
		const controller = new GameController({ rng: seeded(3), scheduler });
		await controller.boot(
			{ init: () => Promise.reject(new Error("no model")), bestMove: () => Promise.reject(new Error("no model")) },
			"/model.pack"
		);
		const snapshot = controller.getSnapshot();
		expect(snapshot.aiReady).toBe(false);
		expect(snapshot.progress).toBe(0);
		const dir = firstChangingDir(controller.engine, snapshot.board);
		expect(controller.humanMove(dir)).toBe(true);
	});

	it("switching shape starts a fresh game on that shape", () => {
		const scheduler = manualScheduler();
		const controller = new GameController({ rng: seeded(5), scheduler });
		controller.setShape(5, 5);
		scheduler.flush();
		const snapshot = controller.getSnapshot();
		expect(snapshot.H).toBe(5);
		expect(snapshot.W).toBe(5);
		expect(snapshot.score).toBe(0);
		expect(snapshot.moves).toBe(0);
		expect(snapshot.board).toHaveLength(25);
	});

	it("reports game over when the board is full with no merges", () => {
		const scheduler = manualScheduler();
		const controller = new GameController({ rng: seeded(1), scheduler });
		expect(controller.getSnapshot().status).toBe("playing");
		expect(controller.hasMoves()).toBe(true);
	});
});
