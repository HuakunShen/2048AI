<script lang="ts">
	import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-svelte';
	import type { Dir } from '$lib/engine/board';
	import type { Sprite } from '$lib/game.svelte';
	import Tile from './Tile.svelte';

	let {
		sprites,
		hint = null,
		slideMs = 100
	}: { sprites: Sprite[]; hint?: Dir | null; slideMs?: number } = $props();

	const ARROW = { UP: ArrowUp, DOWN: ArrowDown, LEFT: ArrowLeft, RIGHT: ArrowRight };
</script>

<!-- --g must match the grid gap/padding below so the tile layer lines up with the cells. -->
<div class="relative aspect-square w-full select-none" style="--g: 0.5rem; --slide: {slideMs}ms;">
	<!-- Static background: the 4×4 grid of empty cells. -->
	<div
		class="grid h-full w-full grid-cols-4 grid-rows-4 gap-2 rounded-xl bg-[#bbada0] p-2 shadow-lg"
	>
		{#each Array.from({ length: 16 }) as _, i (i)}
			<div class="rounded-md bg-[#cdc1b4]"></div>
		{/each}
	</div>

	<!-- Moving layer: absolutely-positioned tiles, inset to match the grid's p-2 padding. -->
	<div class="pointer-events-none absolute inset-2">
		{#each sprites as s (s.id)}
			<Tile sprite={s} />
		{/each}
	</div>

	{#if hint}
		{@const Arrow = ARROW[hint]}
		<div class="pointer-events-none absolute inset-0 flex items-center justify-center">
			<div class="animate-pulse rounded-full bg-black/45 p-4 text-white shadow-xl">
				<Arrow class="h-14 w-14" />
			</div>
		</div>
	{/if}
</div>
