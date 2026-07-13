<script lang="ts">
	import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp } from 'lucide-svelte';
	import { expToTile, type Board, type Dir } from '$lib/engine/board';
	import Tile from './Tile.svelte';

	let { board, hint = null }: { board: Board; hint?: Dir | null } = $props();

	const ARROW = { UP: ArrowUp, DOWN: ArrowDown, LEFT: ArrowLeft, RIGHT: ArrowRight };
</script>

<div class="relative w-full select-none">
	<div class="grid grid-cols-4 gap-2 rounded-xl bg-[#bbada0] p-2 shadow-lg">
		{#each board as exp, i (i)}
			<Tile value={expToTile(exp)} />
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
