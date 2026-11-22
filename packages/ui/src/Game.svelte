<svelte:options customElement={{ tag: "game-2048", shadow: "none" }} />

<script lang="ts">
	import { onMount } from "svelte";
	import {
		GameController,
		SHAPES,
		webStorage,
		type AiProvider,
		type Dir,
		type GameSnapshot
	} from "@2048/core";
	import Board from "./Board.svelte";
	import Controls from "./Controls.svelte";

	/**
	 * `<game-2048>` — the whole interactive game as one element, shared by the
	 * SvelteKit web app and the Kunkun plugin page.
	 *
	 * Properties (optional; methods are the reliable path because a custom element
	 * cannot receive properties before it is upgraded):
	 *   depth (1|2|3), speed (ms/move), autoplay (boolean)
	 *
	 * Methods:
	 *   attachAi(provider, modelUrl)  load the model in a host-owned Worker
	 *   newGame(), move(dir), hint(), aiStep()
	 *   setDepth(n), setSpeed(ms), startAuto(), stopAuto()
	 *
	 * Events (CustomEvent, detail documented at each dispatch):
	 *   score, game-over, ai-state, hint
	 */
	let {
		depth = 2,
		speed = 140,
		autoplay = false
	}: { depth?: number; speed?: number; autoplay?: boolean } = $props();

	const controller = new GameController({
		...(typeof localStorage === "undefined" ? {} : { storage: webStorage(localStorage) }),
	});
	let snapshot = $state<GameSnapshot>(controller.getSnapshot());

	function emit(name: string, detail: unknown): void {
		$host().dispatchEvent(new CustomEvent(name, { detail, bubbles: true }));
	}

	let previous = controller.getSnapshot();

	function publish(next: GameSnapshot): void {
		snapshot = next;
		if (
			next.score !== previous.score ||
			next.moves !== previous.moves ||
			next.maxTile !== previous.maxTile ||
			next.best !== previous.best
		) {
			emit("score", {
				score: next.score,
				best: next.best,
				moves: next.moves,
				maxTile: next.maxTile,
			});
		}
		if (next.status === "over" && previous.status !== "over") {
			emit("game-over", {
				score: next.score,
				best: next.best,
				moves: next.moves,
				maxTile: next.maxTile,
			});
		}
		if (
			next.aiReady !== previous.aiReady ||
			next.thinking !== previous.thinking ||
			next.auto !== previous.auto ||
			next.progress !== previous.progress
		) {
			emit("ai-state", {
				ready: next.aiReady,
				progress: next.progress,
				error: next.aiError,
				thinking: next.thinking,
				auto: next.auto,
				level: next.level,
			});
		}
		previous = next;
	}

	// ── public API (exported functions become methods on the element) ──────────
	export function attachAi(provider: AiProvider, modelUrl: string): Promise<void> {
		return controller.boot(provider, modelUrl);
	}
	export function newGame(): void {
		controller.newGame();
	}
	export function move(dir: Dir): boolean {
		return controller.humanMove(dir);
	}
	export function hint(): Promise<void> {
		return controller.hint();
	}
	export function aiStep(): Promise<boolean> {
		return controller.aiStep();
	}
	export function setDepth(level: number): void {
		controller.setLevel(level);
	}
	export function setSpeed(speedMs: number): void {
		controller.setSpeed(speedMs);
	}
	export function startAuto(): void {
		if (!controller.getSnapshot().auto) controller.toggleAuto();
	}
	export function stopAuto(): void {
		if (controller.getSnapshot().auto) controller.toggleAuto();
	}
	export function setShape(H: number, W: number): void {
		controller.setShape(H, W);
	}

	// ── props → controller ─────────────────────────────────────────────────────
	$effect(() => controller.setLevel(depth));
	$effect(() => controller.setSpeed(speed));
	$effect(() => {
		if (autoplay) controller.toggleAuto();
	});

	// ── lifecycle ──────────────────────────────────────────────────────────────
	const KEYS: Record<string, Dir> = {
		ArrowUp: "UP",
		ArrowDown: "DOWN",
		ArrowLeft: "LEFT",
		ArrowRight: "RIGHT",
	};

	onMount(() => {
		const unsubscribe = controller.subscribe(publish);
		publish(controller.getSnapshot());

		function onKey(event: KeyboardEvent): void {
			const target = event.target;
			if (target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
			const dir = KEYS[event.key];
			if (dir !== undefined) {
				event.preventDefault();
				controller.humanMove(dir);
				return;
			}
			if (event.key === "r" || event.key === "R") controller.newGame();
		}

		window.addEventListener("keydown", onKey);

		const observer = new ResizeObserver((entries) => {
			const rect = entries[0]?.contentRect;
			if (rect === undefined) return;
			freeWidth = rect.width;
			freeHeight = rect.height;
		});
		if (boardArea !== undefined) observer.observe(boardArea);

		return () => {
			observer.disconnect();
			window.removeEventListener("keydown", onKey);
			unsubscribe();
			controller.dispose();
		};
	});

	// ── touch swipe ────────────────────────────────────────────────────────────
	let startX = 0;
	let startY = 0;
	function onTouchStart(event: TouchEvent): void {
		startX = event.touches[0]!.clientX;
		startY = event.touches[0]!.clientY;
	}
	function onTouchEnd(event: TouchEvent): void {
		const dx = event.changedTouches[0]!.clientX - startX;
		const dy = event.changedTouches[0]!.clientY - startY;
		if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
		if (Math.abs(dx) > Math.abs(dy)) controller.humanMove(dx > 0 ? "RIGHT" : "LEFT");
		else controller.humanMove(dy > 0 ? "DOWN" : "UP");
	}

	// Measure the space the chrome leaves, then size the board from the largest
	// square *cell* that fits. Padding and gap are equal on both axes, so scaling
	// a W×H box would leave cells slightly rectangular.
	let boardArea = $state<HTMLDivElement | undefined>(undefined);
	let freeHeight = $state(0);
	let freeWidth = $state(0);
	const PAD = 8;
	const GAP = 8;
	const cell = $derived(
		// Whole pixels: a fractional cell is rounded differently per axis by the
		// grid, which is exactly the 0.4 px difference this avoids.
		Math.floor(
			Math.max(
				0,
				Math.min(
					(freeWidth - 2 * PAD - (snapshot.W - 1) * GAP) / snapshot.W,
					(freeHeight - 2 * PAD - (snapshot.H - 1) * GAP) / snapshot.H
				)
			)
		)
	);
	const boardWidth = $derived(cell > 0 ? snapshot.W * cell + (snapshot.W - 1) * GAP + 2 * PAD : 0);
	const boardHeight = $derived(cell > 0 ? snapshot.H * cell + (snapshot.H - 1) * GAP + 2 * PAD : 0);

	const onNew = () => controller.newGame();
	const onToggleAuto = () => controller.toggleAuto();
	const onStep = () => void controller.aiStep();
	const onHint = () => void controller.hint();
	const onLevel = (level: number) => controller.setLevel(level);
	const onSpeed = (speedMs: number) => controller.setSpeed(speedMs);
</script>

<div class="game">
	<div class="stats">
		<div class="stat">
			<span class="stat-label">Score</span>
			<span class="stat-value">{snapshot.score}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Best</span>
			<span class="stat-value">{snapshot.best}</span>
		</div>
	</div>

	<div class="field">
		<span class="muted small">Board size · one AI plays them all</span>
		<div class="shapes">
			{#each SHAPES as shape (shape.label)}
				<button
					class="chip"
					class:active={snapshot.H === shape.H && snapshot.W === shape.W}
					type="button"
					onclick={() => controller.setShape(shape.H, shape.W)}>{shape.label}</button
				>
			{/each}
		</div>
	</div>

	<div
		class="board-area"
		role="application"
		aria-label="2048 board"
		bind:this={boardArea}
		ontouchstart={onTouchStart}
		ontouchend={onTouchEnd}
	>
		<Board
			sprites={snapshot.sprites}
			H={snapshot.H}
			W={snapshot.W}
			hint={snapshot.hintDir}
			slideMs={snapshot.slideMs}
			width={boardWidth}
			height={boardHeight}
		/>
	</div>

	<div class="status">
		<span class="badge">Max&nbsp;{snapshot.maxTile}</span>
		<span class="muted small num">{snapshot.moves} moves{#if snapshot.thinking}&nbsp;· thinking…{/if}</span>
		{#if snapshot.status === "over"}
			<span class="badge danger">Game over</span>
		{/if}
	</div>

	<Controls {snapshot} {onNew} {onToggleAuto} {onStep} {onHint} {onLevel} {onSpeed} />

	{#if snapshot.status === "over"}
		<div class="over">
			<p class="over-title">Game over</p>
			<p class="muted small">
				Max tile {snapshot.maxTile} · Score {snapshot.score} · {snapshot.moves} moves
			</p>
			<button class="chip active" type="button" onclick={() => controller.newGame()}>New game</button>
		</div>
	{/if}

	<p class="muted tiny footer">
		Arrow keys or swipe to play · press <kbd>r</kbd> to restart
	</p>
</div>

<style>
	.game {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		height: 100%;
		min-height: 0;
		color: var(--foreground, CanvasText);
		font-family: var(--font-sans, system-ui);
		font-size: var(--font-size-base, 1rem);
	}
	.muted {
		color: var(--muted-foreground, GrayText);
	}
	.small {
		font-size: 0.75rem;
	}
	.tiny {
		font-size: 0.6875rem;
	}
	.num {
		font-variant-numeric: tabular-nums;
	}
	.stats {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem;
	}
	.stat {
		display: flex;
		flex-direction: column;
		gap: 0;
		padding: 0.25rem 0.5rem;
		border-radius: var(--radius, 6px);
		background: var(--muted, rgb(127 127 127 / 0.15));
	}
	.stat-label {
		font-size: 0.625rem;
		font-weight: 500;
		letter-spacing: 0.05em;
		text-transform: uppercase;
		color: var(--muted-foreground, GrayText);
	}
	.stat-value {
		font-size: 0.9375rem;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}
	.shapes {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	.chip {
		height: 1.5rem;
		padding: 0 0.5rem;
		border-radius: var(--radius, 6px);
		border: 1px solid var(--border, GrayText);
		background: transparent;
		color: var(--foreground, CanvasText);
		font: inherit;
		font-size: 0.75rem;
		cursor: pointer;
	}
	.chip.active {
		border-color: transparent;
		background: var(--primary, CanvasText);
		color: var(--primary-foreground, Canvas);
	}
	.board-area {
		/* The board takes whatever the chrome leaves. `flex-basis: 0` plus
		   `overflow: hidden` keeps the box independent of the board it contains:
		   otherwise measuring and resizing feed each other and the board spills
		   over the controls. */
		flex: 1 1 0;
		/* A floor, so a very short window still shows a usable square board and
		   the page scrolls instead of collapsing or distorting the cells. */
		min-height: 10rem;
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
		touch-action: none;
	}
	.status {
		font-size: 0.75rem;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	.badge {
		padding: 0.125rem 0.5rem;
		border-radius: var(--radius, 6px);
		background: var(--secondary, rgb(127 127 127 / 0.2));
		color: var(--secondary-foreground, CanvasText);
		font-size: 0.75rem;
	}
	.badge.danger {
		background: var(--destructive, #b91c1c);
		color: #fff;
	}
	.over {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.5rem;
		padding: 1rem;
		border: 1px solid var(--border, GrayText);
		border-radius: calc(var(--radius, 6px) + 2px);
		background: var(--card, transparent);
		text-align: center;
	}
	.over-title {
		margin: 0;
		font-weight: 600;
	}
	.footer {
		margin: 0;
		text-align: center;
	}
</style>
