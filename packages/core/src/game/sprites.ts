/**
 * The sprite model behind the tile animation.
 *
 * Every rendered tile has a stable `id` that survives moves, so the browser can
 * transition one element from cell to cell instead of re-creating the grid. A
 * move is two-phase: phase 1 slides every existing sprite to its destination
 * (state committed immediately), phase 2 collapses merged pairs and adds the
 * freshly spawned tile. `pop` is bumped whenever a sprite should replay its
 * spawn or merge keyframe.
 *
 * Pure functions over plain objects: no DOM, no framework, no timers.
 */
import type { Board, Slide } from "../engine";

/** A rendered tile with a stable identity that persists across moves (for animation). */
export interface Sprite {
	id: number;
	exp: number;
	index: number; // board cell (row*W + col)
	pop: number;
	popKind: "spawn" | "merge" | null;
}

/**
 * Phase 1: move every sprite that has a slide to its destination cell.
 * Returns the cells that received more than one sprite — those merge in phase 2.
 * Mutates the sprites' `index`, which is what drives the CSS transition.
 */
export function slideSprites(sprites: Sprite[], slides: readonly Slide[]): Set<number> {
	const byCell = new Map<number, Sprite>();
	for (const sprite of sprites) byCell.set(sprite.index, sprite);

	const arrivals = new Map<number, number>();
	for (const slide of slides) {
		const sprite = byCell.get(slide.from);
		if (sprite !== undefined) sprite.index = slide.to;
		arrivals.set(slide.to, (arrivals.get(slide.to) ?? 0) + 1);
	}

	const merged = new Set<number>();
	for (const [cell, count] of arrivals) if (count >= 2) merged.add(cell);
	return merged;
}

export interface FinalizeSpritesOptions {
	sprites: readonly Sprite[];
	board: Board;
	mergedCells: ReadonlySet<number>;
	/** Cell the engine spawned on, or -1 when the board was full. */
	spawned: number;
	allocId: () => number;
	allocPop: () => number;
}

/**
 * Phase 2: one sprite per cell, exponents refreshed from the committed board,
 * merged cells replaying their pop, and the spawned tile appended.
 */
export function finalizeSprites(options: FinalizeSpritesOptions): Sprite[] {
	const { board, mergedCells, spawned, allocId, allocPop } = options;
	const seen = new Set<number>();
	const kept: Sprite[] = [];

	for (const sprite of options.sprites) {
		if (seen.has(sprite.index)) continue;
		seen.add(sprite.index);
		sprite.exp = board[sprite.index];
		if (mergedCells.has(sprite.index)) {
			sprite.popKind = "merge";
			sprite.pop = allocPop();
		}
		kept.push(sprite);
	}

	if (spawned >= 0) {
		kept.push({
			id: allocId(),
			exp: board[spawned],
			index: spawned,
			pop: allocPop(),
			popKind: "spawn",
		});
	}
	return kept;
}

/** Fresh sprite list for a board, every tile replaying its spawn. */
export function spritesFromBoard(
	board: Board,
	allocId: () => number,
	allocPop: () => number
): Sprite[] {
	const sprites: Sprite[] = [];
	for (let index = 0; index < board.length; index += 1) {
		const exp = board[index];
		if (exp !== 0) {
			sprites.push({ id: allocId(), exp, index, pop: allocPop(), popKind: "spawn" });
		}
	}
	return sprites;
}
