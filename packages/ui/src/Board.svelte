<script lang="ts">
	import type { Dir, Sprite } from "@2048/core";
	import Tile from "./Tile.svelte";

	let {
		sprites,
		H = 4,
		W = 4,
		hint = null,
		slideMs = 100,
		width = 0,
		height = 0
	}: {
		sprites: Sprite[];
		H?: number;
		W?: number;
		hint?: Dir | null;
		slideMs?: number;
		width?: number;
		height?: number;
	} = $props();

	/**
	 * The host measures the free space and passes box dimensions whose *cells* are
	 * square: padding and gap are the same on both axes, so a W×H board is not a
	 * scaled square. Until the first measurement, fall back to a width-driven box.
	 */
	const box = $derived(
		width > 0 && height > 0 ? `width: ${width}px; height: ${height}px;` : 'width: 100%;'
	);

	const ARROW: Record<Dir, string> = { UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→" };
</script>

<!-- --g must match the grid gap/padding so the tile layer lines up with the cells.
     --cols/--rows drive the generic tile sizing; aspect-ratio keeps cells square. -->
<div
	class="board"
	style="--g: 0.5rem; --slide: {slideMs}ms; --cols: {W}; --rows: {H}; {box}"
>
	<!-- Static background: the H×W grid of empty cells. -->
	<div
		class="grid"
		style="grid-template-columns: repeat({W}, minmax(0, 1fr)); grid-template-rows: repeat({H}, minmax(0, 1fr));"
	>
		{#each Array.from({ length: H * W }) as _, i (i)}
			<div class="cell"></div>
		{/each}
	</div>

	<!-- Moving layer: absolutely-positioned tiles, inset to match the grid's padding. -->
	<div class="layer">
		{#each sprites as s (s.id)}
			<Tile sprite={s} cols={W} />
		{/each}
	</div>

	{#if hint}
		<div class="hint">
			<span class="hint-arrow">{ARROW[hint]}</span>
		</div>
	{/if}
</div>

<style>
	.board {
		position: relative;
		max-width: 100%;
		aspect-ratio: var(--cols) / var(--rows);
		user-select: none;
		container-type: inline-size;
	}
	.grid {
		display: grid;
		gap: var(--g);
		width: 100%;
		height: 100%;
		padding: var(--g);
		box-sizing: border-box;
		border-radius: var(--radius, 12px);
		background: #bbada0;
	}
	.cell {
		border-radius: calc(var(--radius, 6px) - 2px);
		background: #cdc1b4;
	}
	.layer {
		position: absolute;
		inset: var(--g);
		pointer-events: none;
	}
	.hint {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		pointer-events: none;
	}
	.hint-arrow {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 4rem;
		height: 4rem;
		border-radius: 9999px;
		background: rgb(0 0 0 / 0.45);
		color: #fff;
		font-size: 2rem;
		line-height: 1;
		box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.3);
		animation: t2048-pulse 1.2s ease-in-out infinite;
	}
	@keyframes t2048-pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.45;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.hint-arrow {
			animation: none;
		}
	}
</style>
