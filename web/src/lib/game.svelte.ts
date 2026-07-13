/**
 * Reactive game controller (Svelte 5 runes). Owns the board/score state, the
 * shared engine calls, and the AI worker handle. Human moves and AI moves go
 * through the same `applyDir` path.
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
	spawn,
	type Board,
	type Dir
} from './engine/board';
import type { AiAPI, DepthCfg } from './ai/client';

export type Status = 'playing' | 'over';

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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class Game {
	board = $state<Board>(initBoard());
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

	maxTile = $derived(expToTile(maxExp(this.board)));

	/** Browser-only: spin up the AI worker and stream weight-load progress. */
	async boot(): Promise<void> {
		if (!browser || this.ai) return;
		const { createAi } = await import('./ai/client');
		this.ai = createAi();
		await this.ai.init((p) => {
			this.progress = p;
		});
		this.aiReady = true;
	}

	newGame(): void {
		this.stopAuto();
		this.board = initBoard();
		this.score = 0;
		this.moves = 0;
		this.won = false;
		this.hintDir = null;
		this.status = 'playing';
	}

	/** Apply a direction (shared by human + AI). Returns whether the board moved. */
	private applyDir(dir: Dir): boolean {
		if (this.status !== 'playing') return false;
		const { after, reward, changed } = move(this.board, dir);
		if (!changed) return false;
		spawn(after);
		this.board = after; // new reference → reactive
		this.score += reward;
		if (this.score > this.best) this.best = this.score;
		this.moves++;
		this.hintDir = null;
		if (!this.won && hasWon(after)) {
			this.won = true;
			this.onWin?.();
		}
		if (isDone(after)) this.status = 'over';
		return true;
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
