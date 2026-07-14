<script lang="ts">
	import { expToTile } from '$lib/engine/board';
	import { tileBg, tileFg, tileFontClass } from '$lib/theme';
	import type { Sprite } from '$lib/game.svelte';

	let { sprite, cols = 4 }: { sprite: Sprite; cols?: number } = $props();

	const value = $derived(expToTile(sprite.exp));
	const row = $derived(Math.floor(sprite.index / cols));
	const col = $derived(sprite.index % cols);
	const anim = $derived(
		sprite.popKind === 'merge' ? 't2048-merge' : sprite.popKind === 'spawn' ? 't2048-spawn' : ''
	);
</script>

<!-- Persistent element: its `transform` (driven by --r/--c) transitions → the tile slides. -->
<div class="t2048-tile" style="--r:{row}; --c:{col};">
	<!-- Re-keyed on `pop` so spawn/merge replay their keyframe pop; plain slides don't remount. -->
	{#key sprite.pop}
		<div
			class="t2048-inner {anim} {tileFontClass(value)}"
			style="background-color: {tileBg(value)}; color: {tileFg(value)};"
		>
			{value}
		</div>
	{/key}
</div>
