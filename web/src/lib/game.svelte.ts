/**
 * Svelte 5 adapter over `@2048/core`'s framework-free controller.
 *
 * The game state, the engine, the sprite model and the AI all live in the core
 * package so the Kunkun plugin can use exactly the same code. This file only
 * bridges the controller's subscribe/getSnapshot contract into runes, which is
 * what the page and the components read.
 */
import {
	DELAY_MAX,
	DELAY_MIN,
	DEFAULT_SPEED_MS,
	GameController,
	SHAPES,
	STRENGTHS,
	createWorkerAi,
	depthCfgFor,
	webStorage,
	type Dir,
	type GameSnapshot,
} from "@2048/core";

export { DELAY_MAX, DELAY_MIN, DEFAULT_SPEED_MS, SHAPES, STRENGTHS, depthCfgFor };
export type { Dir, GameSnapshot, Sprite } from "@2048/core";

export type Status = "playing" | "over";

export class Game {
	private readonly controller: GameController;
	private snapshot = $state<GameSnapshot>(this.controllerSnapshotPlaceholder());

	/** Set by the page; fires once when the 2048 tile is first reached. */
	onWin: (() => void) | null = null;

	private booted = false;
	private aiHandle: { dispose(): void } | null = null;

	constructor() {
		this.controller = new GameController({
			...(typeof localStorage === "undefined" ? {} : { storage: webStorage(localStorage) }),
			onWin: () => this.onWin?.(),
		});
		this.snapshot = this.controller.getSnapshot();
		this.controller.subscribe((snapshot) => {
			this.snapshot = snapshot;
		});
	}

	/** A placeholder so the rune has an initial value before the controller exists. */
	private controllerSnapshotPlaceholder(): GameSnapshot {
		return {
			H: 4, W: 4, board: new Uint8Array(16), sprites: [], score: 0, best: 0, moves: 0,
			status: "playing", won: false, maxTile: 0, slideMs: 100, aiReady: false,
			progress: 0, thinking: false, auto: false, level: 2, speedMs: DEFAULT_SPEED_MS,
			hintDir: null, hasMoves: true,
		};
	}

	get H() { return this.snapshot.H; }
	get W() { return this.snapshot.W; }
	get board() { return this.snapshot.board; }
	get sprites() { return this.snapshot.sprites; }
	get score() { return this.snapshot.score; }
	get best() { return this.snapshot.best; }
	get moves() { return this.snapshot.moves; }
	get status(): Status { return this.snapshot.status; }
	get won() { return this.snapshot.won; }
	get maxTile() { return this.snapshot.maxTile; }
	get slideMs() { return this.snapshot.slideMs; }
	get aiReady() { return this.snapshot.aiReady; }
	get progress() { return this.snapshot.progress; }
	get thinking() { return this.snapshot.thinking; }
	get auto() { return this.snapshot.auto; }
	get level() { return this.snapshot.level; }
	set level(level: number) { this.controller.setLevel(level); }
	get speedMs() { return this.snapshot.speedMs; }
	set speedMs(speedMs: number) { this.controller.setSpeed(speedMs); }
	get hintDir() { return this.snapshot.hintDir; }
	get hasMoves() { return this.snapshot.hasMoves; }

	/** Load the AI model in a Worker. A failure leaves a playable human game. */
	async boot(): Promise<void> {
		if (this.booted) return;
		this.booted = true;
		const worker = new Worker(new URL("./ai-worker.ts", import.meta.url), { type: "module" });
		const ai = createWorkerAi(worker);
		this.aiHandle = ai;
		await this.controller.boot(ai, new URL("model/model.pack", document.baseURI).href);
	}

	setShape(H: number, W: number): void { this.controller.setShape(H, W); }
	newGame(): void { this.controller.newGame(); }
	humanMove(dir: Dir): void { this.controller.humanMove(dir); }
	aiStep(): Promise<boolean> { return this.controller.aiStep(); }
	toggleAuto(): void { this.controller.toggleAuto(); }
	hint(): Promise<void> { return this.controller.hint(); }

	destroy(): void {
		this.controller.dispose();
		this.aiHandle?.dispose();
	}
}
