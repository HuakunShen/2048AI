/**
 * Reactive game controller (Svelte 5 runes). Owns the board/score state, the
 * shared engine calls, and the AI worker handle. Human moves and AI moves go
 * through the same `applyDir` path.
 *
 * Rendering uses a **stable-id sprite model** so tiles can animate: each move is
 * two-phase — phase 1 slides every existing sprite to its destination cell (the
 * board/score state is committed immediately), then a deferred `finalize`
 * collapses merged pairs (pop) and adds the freshly spawned tile (pop). The
 * board itself stays the source of truth for game logic; sprites are the view.
 */
import { browser } from '$app/environment';
import {
	DIRS,
	expToTile,
	hasWon,
	initBoard,
	isDone,
	maxExp,
	move,
	planMove,
	spawnCell,
	type Board,
	type Dir
} from './engine/board';
import type { AiAPI, DepthCfg } from './ai/client';

export type Status = 'playing' | 'over';

/** A rendered tile with a stable identity that persists across moves (for animation). */
export interface Sprite {
	id: number; // stable across moves → the DOM element persists and its transform transitions
	exp: number; // current exponent (value shown)
	index: number; // board cell 0..15 (row*4 + col)
	pop: number; // nonce; bumping it replays the pop animation (via {#key})
	popKind: 'spawn' | 'merge' | null;
}

/** Search strength: 1 = greedy, 2 = depth-2, 3 = depth-3 adaptive endgame. */
export const STRENGTHS = [
	{ level: 1, label: 'Fast', hint: 'greedy (depth 1)' },
	{ level: 2, label: 'Strong', hint: 'expectimax depth 2' },
	{ level: 3, label: 'Max', hint: 'depth 3, adaptive' }
] as const;

const ADAPTIVE: [number, number][] = [
	[3, 5],
	[7, 3],
	[16, 2]
];

export function depthCfgFor(level: number): DepthCfg {
	if (level <= 1) return { depth: 1 };
	if (level === 2) return { depth: 2 };
	return { depth: 3, adaptive: ADAPTIVE };
}

/** Baseline slide duration (ms) for manual play; clamped tighter during fast auto-play. */
const SLIDE_MS = 100;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class Game {
	board = $state<Board>(initBoard());
	sprites = $state<Sprite[]>([]);
	score = $state(0);
	best = $state(0);
	moves = $state(0);
	status = $state<Status>('playing');
	won = $state(false);

	aiReady = $state(false);
	progress = $state(0); // weight-download progress 0..1
	thinking = $state(false);

	auto = $state(false);
	level = $state(2);
	speedMs = $state(140);
	hintDir = $state<Dir | null>(null);

	/** Fired once when a game first reaches 2048. Set by the page for a toast. */
	onWin: (() => void) | null = null;

	private ai: AiAPI | null = null;
	private nextId = 0;
	private popSeq = 0;
	private pendingTimer: ReturnType<typeof setTimeout> | null = null;
	private pendingFinalize: (() => void) | null = null;

	maxTile = $derived(expToTile(maxExp(this.board)));

	/** Slide long enough to read, but never longer than an auto-play tick. */
	slideMs = $derived(this.auto ? Math.min(SLIDE_MS, Math.max(45, this.speedMs)) : SLIDE_MS);

	constructor() {
		this.resetSprites();
	}

	/** Rebuild the sprite list from `board`, one popping tile per occupied cell. */
	private resetSprites(): void {
		const arr: Sprite[] = [];
		for (let i = 0; i < 16; i++) {
			if (this.board[i] !== 0)
				arr.push({ id: this.nextId++, exp: this.board[i], index: i, pop: ++this.popSeq, popKind: 'spawn' });
		}
		this.sprites = arr;
	}

	/** Browser-only: spin up the AI worker and stream weight-load progress. */
	async boot(): Promise<void> {
		if (!browser || this.ai) return;
		const { createAi } = await import('./ai/client');
		this.ai = createAi();
		// Resolve the model URL on the main thread (base-path aware) — the worker
		// can't, since its own chunk path skews a relative BASE_URL.
		const modelBase = new URL('model/', document.baseURI).href;
		await this.ai.init(modelBase, (p) => {
			this.progress = p;
		});
		this.aiReady = true;
	}

	newGame(): void {
		this.stopAuto();
		this.cancelPending();
		this.board = initBoard();
		this.score = 0;
		this.moves = 0;
		this.won = false;
		this.hintDir = null;
		this.status = 'playing';
		this.resetSprites();
	}

	/** Apply a direction (shared by human + AI). Returns whether the board moved. */
	private applyDir(dir: Dir): boolean {
		if (this.status !== 'playing') return false;
		this.flushPending(); // settle the previous move's sprites before starting a new slide

		const plan = planMove(this.board, dir);
		if (!plan.changed) return false;

		// Phase 1 — slide every existing sprite to its destination cell.
		const byCell = new Map<number, Sprite>();
		for (const s of this.sprites) byCell.set(s.index, s);
		const toCount = new Map<number, number>();
		for (const sl of plan.slides) {
			const s = byCell.get(sl.from);
			if (s) s.index = sl.to; // reactive → CSS `transform` transition slides the tile
			toCount.set(sl.to, (toCount.get(sl.to) ?? 0) + 1);
		}
		const mergedCells = new Set<number>();
		for (const [cell, n] of toCount) if (n >= 2) mergedCells.add(cell);

		// Commit game state now (logic must not wait on the animation).
		const after = plan.after;
		const spawned = spawnCell(after);
		this.board = after;
		this.score += plan.reward;
		if (this.score > this.best) this.best = this.score;
		this.moves++;
		this.hintDir = null;
		if (!this.won && hasWon(after)) {
			this.won = true;
			this.onWin?.();
		}
		if (isDone(after)) this.status = 'over';

		// Phase 2 (deferred) — drop absorbed ghosts, pop merged tiles + the spawn.
		this.pendingFinalize = () => {
			const seen = new Set<number>();
			const kept: Sprite[] = [];
			for (const s of this.sprites) {
				if (seen.has(s.index)) continue; // second tile of a merge — absorbed
				seen.add(s.index);
				kept.push(s);
			}
			for (const s of kept) {
				s.exp = this.board[s.index];
				if (mergedCells.has(s.index)) {
					s.popKind = 'merge';
					s.pop = ++this.popSeq; // new value + replay the merge pop
				}
			}
			if (spawned >= 0) {
				kept.push({
					id: this.nextId++,
					exp: this.board[spawned],
					index: spawned,
					pop: ++this.popSeq,
					popKind: 'spawn'
				});
			}
			this.sprites = kept;
			this.pendingFinalize = null;
			this.pendingTimer = null;
		};
		this.pendingTimer = setTimeout(this.pendingFinalize, this.slideMs);
		return true;
	}

	/** Run any deferred finalize immediately (before the next move / on demand). */
	private flushPending(): void {
		if (this.pendingTimer) {
			clearTimeout(this.pendingTimer);
			this.pendingTimer = null;
		}
		const f = this.pendingFinalize;
		if (f) {
			this.pendingFinalize = null;
			f();
		}
	}

	/** Drop a deferred finalize without running it (sprites are being rebuilt). */
	private cancelPending(): void {
		if (this.pendingTimer) {
			clearTimeout(this.pendingTimer);
			this.pendingTimer = null;
		}
		this.pendingFinalize = null;
	}

	humanMove(dir: Dir): void {
		if (this.auto) return; // ignore manual moves while auto-playing
		this.applyDir(dir);
	}

	/** One AI move. Returns whether it moved. */
	async aiStep(): Promise<boolean> {
		if (!this.ai || !this.aiReady || this.status !== 'playing') return false;
		this.thinking = true;
		try {
			const { dir } = await this.ai.bestMove(Array.from(this.board), depthCfgFor(this.level));
			if (!dir) {
				this.status = 'over';
				return false;
			}
			return this.applyDir(dir);
		} finally {
			this.thinking = false;
		}
	}

	toggleAuto(): void {
		if (this.auto) {
			this.stopAuto();
		} else if (this.aiReady && this.status === 'playing') {
			this.auto = true;
			void this.autoLoop();
		}
	}

	private stopAuto(): void {
		this.auto = false;
	}

	private async autoLoop(): Promise<void> {
		while (this.auto && this.status === 'playing') {
			const moved = await this.aiStep();
			if (!moved) break;
			await sleep(this.speedMs);
		}
		this.auto = false;
	}

	async hint(): Promise<void> {
		if (!this.ai || !this.aiReady || this.status !== 'playing') return;
		this.thinking = true;
		try {
			const { dir } = await this.ai.bestMove(Array.from(this.board), depthCfgFor(this.level));
			this.hintDir = dir;
		} finally {
			this.thinking = false;
		}
	}

	/** True if at least one legal move remains (used defensively). */
	get hasMoves(): boolean {
		return DIRS.some((d) => move(this.board, d).changed);
	}
}
