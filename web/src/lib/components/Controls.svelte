<script lang="ts">
	import { Button, Progress, Slider } from '@kksh/svelte5';
	import { Gauge, Lightbulb, Pause, Play, RotateCcw, StepForward } from 'lucide-svelte';
	import {
		DEFAULT_SPEED_MS,
		DELAY_MAX,
		DELAY_MIN,
		STRENGTHS,
		type Game
	} from '$lib/game.svelte';

	let { game }: { game: Game } = $props();

	const playable = $derived(game.aiReady && game.status === 'playing');

	// The engine wants a *delay* (ms/move), but a "speed" slider must run
	// slow→fast left→right. So the slider tracks speed and we mirror it onto the
	// delay: the right end (DELAY_MAX) maps to the smallest delay (DELAY_MIN).
	const toDelay = (s: number) => DELAY_MIN + DELAY_MAX - s;

	let speed = $state(toDelay(DEFAULT_SPEED_MS));
	$effect(() => {
		game.speedMs = toDelay(speed);
	});
</script>

<div class="flex w-full flex-col gap-4">
	{#if !game.aiReady}
		<div class="flex flex-col gap-1.5">
			<div class="flex justify-between text-xs text-muted-foreground">
				<span>Downloading AI model…</span>
				<span class="tabular-nums">{Math.round(game.progress * 100)}%</span>
			</div>
			<Progress value={game.progress * 100} />
		</div>
	{/if}

	<div class="grid grid-cols-2 gap-2">
		<Button class="h-11" variant="secondary" onclick={() => game.newGame()}>
			<RotateCcw class="size-4" /> New
		</Button>
		<Button class="h-11" onclick={() => game.toggleAuto()} disabled={!playable}>
			{#if game.auto}
				<Pause class="size-4" /> Pause
			{:else}
				<Play class="size-4" /> Auto-play
			{/if}
		</Button>
		<Button class="h-11" variant="outline" onclick={() => game.aiStep()} disabled={!playable || game.auto}>
			<StepForward class="size-4" /> Step
		</Button>
		<Button class="h-11" variant="outline" onclick={() => game.hint()} disabled={!playable || game.auto}>
			<Lightbulb class="size-4" /> Hint
		</Button>
	</div>

	<div class="flex flex-col gap-1.5">
		<span class="text-xs text-muted-foreground">AI strength</span>
		<div class="grid grid-cols-3 gap-2">
			{#each STRENGTHS as s (s.level)}
				<Button
					class="h-10"
					variant={game.level === s.level ? 'default' : 'outline'}
					title={s.hint}
					onclick={() => (game.level = s.level)}
				>
					{s.label}
				</Button>
			{/each}
		</div>
		<span class="text-[11px] text-muted-foreground">
			{STRENGTHS.find((s) => s.level === game.level)?.hint}
		</span>
	</div>

	<div class="flex flex-col gap-2">
		<div class="flex items-center justify-between text-xs text-muted-foreground">
			<span class="flex items-center gap-1.5"><Gauge class="size-3.5" /> Auto-play speed</span>
			<span class="tabular-nums">{game.speedMs} ms/move</span>
		</div>
		<Slider type="single" min={DELAY_MIN} max={DELAY_MAX} step={20} bind:value={speed} />
	</div>
</div>
