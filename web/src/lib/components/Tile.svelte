<script lang="ts">
	import { scale } from 'svelte/transition';
	import { tileBg, tileFg, tileFontClass } from '$lib/theme';

	let { value }: { value: number } = $props();
</script>

<div class="relative aspect-square rounded-md" style="background-color: {tileBg(0)}">
	{#if value > 0}
		{#key value}
			<div
				in:scale={{ duration: 110, start: 0.55 }}
				class="absolute inset-0 flex items-center justify-center rounded-md font-bold tabular-nums {tileFontClass(
					value
				)}"
				style="background-color: {tileBg(value)}; color: {tileFg(value)}"
			>
				{value}
			</div>
		{/key}
	{/if}
</div>
