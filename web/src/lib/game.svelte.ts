/**
 * Reactive game controller (Svelte 5 runes) for a variable H×W board. Owns the
 * board/score state, a shape-bound {@link Engine}, and the AI worker handle.
 * Human and AI moves share the same `applyDir` path.
 *
 * Rendering uses a **stable-id sprite model** so tiles can animate: each move is
 * two-phase — phase 1 slides every existing sprite to its destination cell (state
 * committed immediately), then a deferred `finalize` collapses merged pairs (pop)
 * and adds the freshly spawned tile.
 */
import { browser } from '$app/environment';
import { DIRS, Engine, expToTile, type Board, type Dir } from './engine/board';
import type { AiAPI, DepthCfg } from './ai/client';

export type Status = 'playing' | 'over';

/** Selectable board shapes (min dimension ≥ 4 plays best; 3×N is intentionally hard). */
export const SHAPES = [
	{ H: 4, W: 4, label: '4×4' },
	{ H: 5, W: 5, label: '5×5' },
	{ H: 4, W: 5, label: '4×5' },
	{ H: 5, W: 4, label: '5×4' },
	{ H: 6, W: 6, label: '6×6' },
	{ H: 3, W: 4, label: '3×4' }
] as const;

/** A rendered tile with a stable identity that persists across moves (for animation). */
export interface Sprite {
	id: number;
	exp: number;
	index: number; // board cell (row*W + col)
	pop: number;
	popKind: 'spawn' | 'merge' | null;
}

/** Search strength: 1 = greedy, 2 = depth-2, 3 = depth-3 adaptive endgame. */
export const STRENGTHS = [
	{ level: 1, label: 'Fast', hint: 'greedy (depth 1)' },
	{ level: 2, label: 'Strong', hint: 'expectimax depth 2' },
	{ level: 3, label: 'Max', hint: 'depth 3, adaptive' }
] as const;

// Adaptive rules by empty-cell *ratio* (works for any board size).
const ADAPTIVE: [number, number][] = [
	[0.15, 4],
	[0.35, 3]
];

export function depthCfgFor(level: number): DepthCfg {
	if (level <= 1) return { depth: 1 };
	if (level === 2) return { depth: 2 };
	return { depth: 3, adaptive: ADAPTIVE, elseDepth: 2, maxChanceCells: 8 };
}

const SLIDE_MS = 100;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class Game {
	H = $state(4);
	W = $state(4);
	engine = new Engine(4, 4);

	board = $state<Board>(new Engine(4, 4).initBoard());
	sprites = $state<Sprite[]>([]);
	score = $state(0);
	best = $state(0);
	moves = $state(0);
	status = $state<Status>('playing');
	won = $state(false);

	aiReady = $state(false);
	progress = $state(0);
	thinking = $state(false);

	auto = $state(false);
	level = $state(2);
	speedMs = $state(140);
	hintDir = $state<Dir | null>(null);

	onWin: (() => void) | null = null;

	private ai: AiAPI | null = null;
	private nextId = 0;
	private popSeq = 0;
	private pendingTimer: ReturnType<typeof setTimeout> | null = null;
	private pendingFinalize: (() => void) | null = null;

	maxTile = $derived(expToTile(this.engine.maxExp(this.board)));
	slideMs = $derived(this.auto ? Math.min(SLIDE_MS, Math.max(45, this.speedMs)) : SLIDE_MS);

	constructor() {
		this.board = this.engine.initBoard();
		this.resetSprites();
	}

	private resetSprites(): void {
		const arr: Sprite[] = [];
		for (let i = 0; i < this.engine.nCells; i++) {
			if (this.board[i] !== 0)
				arr.push({ id: this.nextId++, exp: this.board[i], index: i, pop: ++this.popSeq, popKind: 'spawn' });
		}
		this.sprites = arr;
	}

	async boot(): Promise<void> {
		if (!browser || this.ai) return;
		const { createAi } = await import('./ai/client');
		this.ai = createAi();
		const modelBase = new URL('model/', document.baseURI).href;
		await this.ai.init(modelBase, (p) => {
			this.progress = p;
		});
		this.aiReady = true;
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
		this.newGame();
	}

	newGame(): void {
		this.stopAuto();
		this.cancelPending();
		this.board = this.engine.initBoard();
		this.score = 0;
		this.moves = 0;
		this.won = false;
		this.hintDir = null;
		this.status = 'playing';
		this.resetSprites();
	}

	private applyDir(dir: Dir): boolean {
		if (this.status !== 'playing') return false;
		this.flushPending();

		const plan = this.engine.planMove(this.board, dir);
		if (!plan.changed) return false;

		const byCell = new Map<number, Sprite>();
		for (const s of this.sprites) byCell.set(s.index, s);
		const toCount = new Map<number, number>();
		for (const sl of plan.slides) {
			const s = byCell.get(sl.from);
			if (s) s.index = sl.to;
			toCount.set(sl.to, (toCount.get(sl.to) ?? 0) + 1);
		}
		const mergedCells = new Set<number>();
		for (const [cell, n] of toCount) if (n >= 2) mergedCells.add(cell);

		const after = plan.after;
		const spawned = this.engine.spawnCell(after);
		this.board = after;
		this.score += plan.reward;
		if (this.score > this.best) this.best = this.score;
		this.moves++;
		this.hintDir = null;
		if (!this.won && this.engine.hasWon(after)) {
			this.won = true;
			this.onWin?.();
		}
		if (this.engine.isDone(after)) this.status = 'over';

		this.pendingFinalize = () => {
			const seen = new Set<number>();
			const kept: Sprite[] = [];
			for (const s of this.sprites) {
				if (seen.has(s.index)) continue;
				seen.add(s.index);
				kept.push(s);
			}
			for (const s of kept) {
				s.exp = this.board[s.index];
				if (mergedCells.has(s.index)) {
					s.popKind = 'merge';
					s.pop = ++this.popSeq;
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

	private cancelPending(): void {
		if (this.pendingTimer) {
			clearTimeout(this.pendingTimer);
			this.pendingTimer = null;
		}
		this.pendingFinalize = null;
	}

	humanMove(dir: Dir): void {
		if (this.auto) return;
		this.applyDir(dir);
	}

	async aiStep(): Promise<boolean> {
		if (!this.ai || !this.aiReady || this.status !== 'playing') return false;
		this.thinking = true;
		try {
			const { dir } = await this.ai.bestMove(
				Array.from(this.board),
				this.H,
				this.W,
				depthCfgFor(this.level)
			);
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
			const { dir } = await this.ai.bestMove(
				Array.from(this.board),
				this.H,
				this.W,
				depthCfgFor(this.level)
			);
			this.hintDir = dir;
		} finally {
			this.thinking = false;
		}
	}

	get hasMoves(): boolean {
		return DIRS.some((d) => this.engine.move(this.board, d).changed);
	}
}
