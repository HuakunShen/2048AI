/**
 * Headless game controller for a variable H×W board.
 *
 * Owns the board/score state, a shape-bound {@link Engine}, the sprite list and
 * (optionally) an AI provider. It is framework-free: no Svelte runes, no React,
 * no DOM. Hosts read {@link GameController.getSnapshot} and re-render when
 * {@link GameController.subscribe} fires — the contract `useSyncExternalStore`
 * expects, and a plain callback everywhere else.
 *
 * Human and AI moves share the same `applyDir` path, and a move is two-phase:
 * the slide is committed immediately, then a deferred finalise collapses merged
 * pairs and adds the spawned tile after `slideMs`. That is what makes the tile
 * animation correct, and it is why the scheduler is injectable.
 */
import { DIRS, Engine, expToTile, type Board, type Dir, type Rng } from "../engine";
import type { DepthCfg } from "../ai/expectimax";
import { finalizeSprites, slideSprites, spritesFromBoard, type Sprite } from "./sprites";
import { BEST_KEY, SHAPE_KEY, memoryStorage, type GameStorage } from "./storage";

export type Status = "playing" | "over";

/** Selectable board shapes (min dimension ≥ 4 plays best; 3×N is intentionally hard). */
export const SHAPES = [
	{ H: 4, W: 4, label: "4×4" },
	{ H: 5, W: 5, label: "5×5" },
	{ H: 4, W: 5, label: "4×5" },
	{ H: 5, W: 4, label: "5×4" },
	{ H: 6, W: 6, label: "6×6" },
	{ H: 3, W: 4, label: "3×4" },
] as const;

/** Search strength: 1 = greedy, 2 = depth-2, 3 = depth-3 adaptive endgame. */
export const STRENGTHS = [
	{ level: 1, label: "Fast", hint: "greedy (depth 1)" },
	{ level: 2, label: "Strong", hint: "expectimax depth 2" },
	{ level: 3, label: "Max", hint: "depth 3, adaptive" },
] as const;

// Adaptive rules by empty-cell *ratio* (works for any board size).
const ADAPTIVE: [number, number][] = [
	[0.15, 4],
	[0.35, 3],
];

export function depthCfgFor(level: number): DepthCfg {
	if (level <= 1) return { depth: 1 };
	if (level === 2) return { depth: 2 };
	return { depth: 3, adaptive: ADAPTIVE, elseDepth: 2, maxChanceCells: 8 };
}

/** How long a slide takes before merged tiles collapse and the spawn appears. */
export const SLIDE_MS = 100;

/** Auto-play delay bounds (ms per move). The UI shows *speed*, mirrored onto this range. */
export const DELAY_MIN = 20;
export const DELAY_MAX = 600;
export const DEFAULT_SPEED_MS = 140;

/** A scheduler, so tests can drive the two-phase commit deterministically. */
export interface Scheduler {
	/** Run `fn` after `ms`; returns a cancel function. */
	after(ms: number, fn: () => void): () => void;
}

const timerScheduler: Scheduler = {
	after(ms, fn) {
		const handle = setTimeout(fn, ms);
		return () => clearTimeout(handle);
	},
};

/** A single AI answer, mirroring the worker's response. */
export interface AiMove {
	dir: Dir | null;
	value: number;
}

/** What the controller needs from an AI; the worker implementation lives in `ai/client`. */
export interface AiProvider {
	init(modelUrl: string, onProgress: (fraction: number) => void): Promise<void>;
	bestMove(board: number[], H: number, W: number, cfg: DepthCfg): Promise<AiMove>;
}

export interface GameSnapshot {
	H: number;
	W: number;
	board: Board;
	sprites: Sprite[];
	score: number;
	best: number;
	moves: number;
	status: Status;
	won: boolean;
	maxTile: number;
	slideMs: number;
	aiReady: boolean;
	progress: number;
	/** Non-null when the model load failed; the human game still works. */
	aiError: string | null;
	thinking: boolean;
	auto: boolean;
	level: number;
	speedMs: number;
	hintDir: Dir | null;
	hasMoves: boolean;
}

export interface GameOptions {
	H?: number;
	W?: number;
	/** Injectable RNG — pass a seeded one to make a game reproducible. */
	rng?: Rng;
	/** Where best/shape are kept. Defaults to a process-lifetime store. */
	storage?: GameStorage;
	scheduler?: Scheduler;
	/** Called once when the 2048 tile is first reached. */
	onWin?: () => void;
}

function readShape(storage: GameStorage): { H: number; W: number } | null {
	const raw = storage.get(SHAPE_KEY);
	if (raw === null) return null;
	const match = /^(\d+)x(\d+)$/.exec(raw);
	if (match === null) return null;
	const H = Number(match[1]);
	const W = Number(match[2]);
	if (!Number.isInteger(H) || !Number.isInteger(W)) return null;
	if (!SHAPES.some((shape) => shape.H === H && shape.W === W)) return null;
	return { H, W };
}

export class GameController {
	H: number;
	W: number;
	engine: Engine;

	private board: Board;
	private sprites: Sprite[];
	private score = 0;
	private best = 0;
	private moves = 0;
	private status: Status = "playing";
	private won = false;

	private aiReady = false;
	private progress = 0;
	private thinking = false;

	private auto = false;
	private level = 2;
	private speedMs = DEFAULT_SPEED_MS;
	private hintDir: Dir | null = null;

	private readonly rng: Rng;
	private readonly storage: GameStorage;
	private readonly scheduler: Scheduler;
	private readonly onWin: (() => void) | undefined;

	private ai: AiProvider | null = null;
	private modelUrl: string | null = null;
	private aiError: string | null = null;
	private nextId = 0;
	private popSeq = 0;
	private cancelFinalize: (() => void) | null = null;
	private finalize: (() => void) | null = null;

	private listeners = new Set<(snapshot: GameSnapshot) => void>();
	private snapshot: GameSnapshot | null = null;
	private autoCancel: (() => void) | null = null;
	private autoResolve: (() => void) | null = null;

	constructor(options: GameOptions = {}) {
		const stored = options.H === undefined || options.W === undefined ? readShape(options.storage ?? memoryStorage()) : null;
		const H = options.H ?? stored?.H ?? 4;
		const W = options.W ?? stored?.W ?? 4;

		this.H = H;
		this.W = W;
		this.engine = new Engine(H, W);
		this.rng = options.rng ?? Math.random;
		this.storage = options.storage ?? memoryStorage();
		this.scheduler = options.scheduler ?? timerScheduler;
		this.onWin = options.onWin;
		this.best = Number.parseInt(this.storage.get(BEST_KEY) ?? "", 10) || 0;

		this.board = this.engine.initBoard(this.rng);
		this.sprites = this.resetSprites();
	}

	// ── reading ───────────────────────────────────────────────────────────────

	getSnapshot(): GameSnapshot {
		if (this.snapshot !== null) return this.snapshot;
		const slideMs = this.auto ? Math.min(SLIDE_MS, Math.max(45, this.speedMs)) : SLIDE_MS;
		this.snapshot = {
			H: this.H,
			W: this.W,
			board: this.board,
			sprites: this.sprites,
			score: this.score,
			best: this.best,
			moves: this.moves,
			status: this.status,
			won: this.won,
			maxTile: expToTile(this.engine.maxExp(this.board)),
			slideMs,
			aiReady: this.aiReady,
			progress: this.progress,
			aiError: this.aiError,
			thinking: this.thinking,
			auto: this.auto,
			level: this.level,
			speedMs: this.speedMs,
			hintDir: this.hintDir,
			hasMoves: this.hasMoves(),
		};
		return this.snapshot;
	}

	subscribe(listener: (snapshot: GameSnapshot) => void): () => void {
		this.listeners.add(listener);
		return () => {
			this.listeners.delete(listener);
		};
	}

	private emit(): void {
		this.snapshot = null;
		const snapshot = this.getSnapshot();
		for (const listener of this.listeners) listener(snapshot);
	}

	hasMoves(): boolean {
		return DIRS.some((dir) => this.engine.move(this.board, dir).changed);
	}

	// ── settings ──────────────────────────────────────────────────────────────

	setLevel(level: number): void {
		this.level = level;
		this.emit();
	}

	setSpeed(speedMs: number): void {
		this.speedMs = Math.min(DELAY_MAX, Math.max(DELAY_MIN, speedMs));
		this.emit();
	}

	/** Switch board shape and start a fresh game on it. */
	setShape(H: number, W: number): void {
		if (H === this.H && W === this.W) {
			this.newGame();
			return;
		}
		this.stopAuto();
		this.cancelPending();
		this.H = H;
		this.W = W;
		this.engine = new Engine(H, W);
		this.storage.set(SHAPE_KEY, `${H}x${W}`);
		this.newGame();
	}

	newGame(): void {
		this.stopAuto();
		this.cancelPending();
		this.board = this.engine.initBoard(this.rng);
		this.score = 0;
		this.moves = 0;
		this.won = false;
		this.hintDir = null;
		this.status = "playing";
		this.sprites = this.resetSprites();
		this.emit();
	}

	private resetSprites(): Sprite[] {
		return spritesFromBoard(
			this.board,
			() => this.nextId++,
			() => ++this.popSeq
		);
	}

	// ── moves ─────────────────────────────────────────────────────────────────

	/** Apply a human move. Ignored while the AI is auto-playing. */
	humanMove(dir: Dir): boolean {
		if (this.auto) return false;
		return this.applyDir(dir);
	}

	private applyDir(dir: Dir): boolean {
		if (this.status !== "playing") return false;
		this.flushPending();

		const plan = this.engine.planMove(this.board, dir);
		if (!plan.changed) return false;

		const mergedCells = slideSprites(this.sprites, plan.slides);
		const after = plan.after;
		const spawned = this.engine.spawnCell(after, this.rng);

		this.board = after;
		this.score += plan.reward;
		if (this.score > this.best) {
			this.best = this.score;
			this.storage.set(BEST_KEY, String(this.best));
		}
		this.moves += 1;
		this.hintDir = null;
		if (!this.won && this.engine.hasWon(after)) {
			this.won = true;
			this.onWin?.();
		}
		if (this.engine.isDone(after)) this.status = "over";

		this.finalize = () => {
			this.sprites = finalizeSprites({
				sprites: this.sprites,
				board: this.board,
				mergedCells,
				spawned,
				allocId: () => this.nextId++,
				allocPop: () => ++this.popSeq,
			});
			this.finalize = null;
			this.cancelFinalize = null;
			this.emit();
		};
		this.cancelFinalize = this.scheduler.after(this.getSnapshot().slideMs, this.finalize);
		this.emit();
		return true;
	}

	private flushPending(): void {
		this.cancelFinalize?.();
		this.cancelFinalize = null;
		const finalize = this.finalize;
		if (finalize !== null) {
			this.finalize = null;
			finalize();
		}
	}

	private cancelPending(): void {
		this.cancelFinalize?.();
		this.cancelFinalize = null;
		this.finalize = null;
	}

	// ── AI ────────────────────────────────────────────────────────────────────

	/** Attach an AI and load its model. A failure leaves a playable human game. */
	async boot(provider: AiProvider, modelUrl: string): Promise<void> {
		if (this.ai !== null) return;
		this.ai = provider;
		this.modelUrl = modelUrl;
		await this.load();
	}

	/** Drop a failed AI and try again; the human game keeps working either way. */
	async retry(): Promise<void> {
		this.ai = null;
		this.aiReady = false;
		this.aiError = null;
		this.progress = 0;
		this.emit();
	}

	/** Re-run the model load on an already-attached provider. */
	async load(): Promise<void> {
		if (this.ai === null || this.aiReady || this.modelUrl === null) return;
		this.aiError = null;
		try {
			await this.ai.init(this.modelUrl, (fraction) => {
				this.progress = fraction;
				this.emit();
			});
			this.aiReady = true;
			this.progress = 1;
		} catch (error) {
			// A missing or corrupt pack is not fatal: the human game keeps working.
			this.aiReady = false;
			this.progress = 0;
			this.aiError = error instanceof Error ? error.message : String(error);
		} finally {
			this.emit();
		}
	}

	async aiStep(): Promise<boolean> {
		if (this.ai === null || !this.aiReady || this.status !== "playing") return false;
		this.thinking = true;
		this.emit();
		try {
			const { dir } = await this.ai.bestMove(
				Array.from(this.board),
				this.H,
				this.W,
				depthCfgFor(this.level)
			);
			if (dir === null) {
				this.status = "over";
				this.emit();
				return false;
			}
			return this.applyDir(dir);
		} finally {
			this.thinking = false;
			this.emit();
		}
	}

	toggleAuto(): void {
		if (this.auto) {
			this.stopAuto();
			return;
		}
		if (!this.aiReady || this.status !== "playing") return;
		this.auto = true;
		this.emit();
		void this.autoLoop();
	}

	private stopAuto(): void {
		this.autoCancel?.();
		this.autoCancel = null;
		// Wake the loop: cancelling the timer alone would leave its await pending forever.
		this.autoResolve?.();
		this.autoResolve = null;
		if (!this.auto) return;
		this.auto = false;
		this.emit();
	}

	private async autoLoop(): Promise<void> {
		while (this.auto && this.status === "playing") {
			const moved = await this.aiStep();
			if (!moved) break;
			await new Promise<void>((resolve) => {
				this.autoResolve = resolve;
				this.autoCancel = this.scheduler.after(this.speedMs, () => {
					this.autoCancel = null;
					this.autoResolve = null;
					resolve();
				});
			});
		}
		this.stopAuto();
	}

	async hint(): Promise<void> {
		if (this.ai === null || !this.aiReady || this.status !== "playing") return;
		this.thinking = true;
		this.emit();
		try {
			const { dir } = await this.ai.bestMove(
				Array.from(this.board),
				this.H,
				this.W,
				depthCfgFor(this.level)
			);
			this.hintDir = dir;
		} finally {
			this.thinking = false;
			this.emit();
		}
	}

	dispose(): void {
		this.stopAuto();
		this.cancelPending();
		this.listeners.clear();
	}
}
