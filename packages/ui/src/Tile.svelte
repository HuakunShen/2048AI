<script lang="ts">
	import { expToTile, type Sprite } from "@2048/core";
	import { tileBg, tileFg, tileSizeClass } from "./palette";

	let { sprite, cols = 4 }: { sprite: Sprite; cols?: number } = $props();

	const value = $derived(expToTile(sprite.exp));
	const row = $derived(Math.floor(sprite.index / cols));
	const col = $derived(sprite.index % cols);
	const anim = $derived(
		sprite.popKind === "merge" ? "t2048-merge" : sprite.popKind === "spawn" ? "t2048-spawn" : ""
	);
</script>

<!-- Persistent element: its `transform` (driven by --r/--c) transitions → the tile slides. -->
<div class="t2048-tile" style="--r:{row}; --c:{col};">
	<!-- Re-keyed on `pop` so spawn/merge replay their keyframe pop; plain slides don't remount. -->
	{#key sprite.pop}
		<div
			class="t2048-inner {anim} {tileSizeClass(value)}"
			style="background-color: {tileBg(value)}; color: {tileFg(value)};"
		>
			{value}
		</div>
	{/key}
</div>

<style>
	.t2048-tile {
		position: absolute;
		top: 0;
		left: 0;
		width: calc((100% - (var(--cols) - 1) * var(--g)) / var(--cols));
		height: calc((100% - (var(--rows) - 1) * var(--g)) / var(--rows));
		transform: translate(calc((100% + var(--g)) * var(--c)), calc((100% + var(--g)) * var(--r)));
		transition: transform var(--slide, 100ms) ease-in-out;
		will-change: transform;
	}
	.t2048-inner {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		border-radius: calc(var(--radius, 6px) - 2px);
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		line-height: 1;
	}
	.size-lg {
		font-size: clamp(1.5rem, 7cqw, 2.25rem);
	}
	.size-md {
		font-size: clamp(1.1rem, 5.5cqw, 1.75rem);
	}
	.size-sm {
		font-size: clamp(0.9rem, 4.5cqw, 1.35rem);
	}
	.size-xs {
		font-size: clamp(0.75rem, 3.6cqw, 1.1rem);
	}
	@keyframes t2048-spawn {
		from {
			transform: scale(0);
			opacity: 0;
		}
		to {
			transform: scale(1);
			opacity: 1;
		}
	}
	@keyframes t2048-merge {
		0% {
			transform: scale(1);
		}
		50% {
			transform: scale(1.18);
		}
		100% {
			transform: scale(1);
		}
	}
	.t2048-spawn {
		animation: t2048-spawn 130ms ease;
	}
	.t2048-merge {
		animation: t2048-merge 150ms ease;
	}
	@media (prefers-reduced-motion: reduce) {
		.t2048-tile {
			transition: none;
		}
		.t2048-spawn,
		.t2048-merge {
			animation: none;
		}
	}
</style>
