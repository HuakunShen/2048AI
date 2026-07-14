<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Badge, Button } from '@kksh/svelte5';
	import { Moon, Sun } from 'lucide-svelte';
	import { toggleMode } from 'mode-watcher';
	import Board from '$lib/components/Board.svelte';
	import Controls from '$lib/components/Controls.svelte';
	import { Game, SHAPES } from '$lib/game.svelte';
	import type { Dir } from '$lib/engine/board';

	const game = new Game();
	game.onWin = () =>
		toast.success('Reached 2048! 🎉', { description: 'Keep going — the AI can hit 4096+.' });

	onMount(() => {
		game.boot();
	});

	const KEYS: Record<string, Dir> = {
		ArrowUp: 'UP',
		ArrowDown: 'DOWN',
		ArrowLeft: 'LEFT',
		ArrowRight: 'RIGHT'
	};

	function onKey(e: KeyboardEvent) {
		if (e.key in KEYS) {
			e.preventDefault();
			game.humanMove(KEYS[e.key]);
		} else if (e.key === 'r' || e.key === 'R') {
			game.newGame();
		}
	}

	// Touch swipe controls.
	let sx = 0;
	let sy = 0;
	function onTouchStart(e: TouchEvent) {
		sx = e.touches[0].clientX;
		sy = e.touches[0].clientY;
	}
	function onTouchEnd(e: TouchEvent) {
		const dx = e.changedTouches[0].clientX - sx;
		const dy = e.changedTouches[0].clientY - sy;
		if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
		if (Math.abs(dx) > Math.abs(dy)) game.humanMove(dx > 0 ? 'RIGHT' : 'LEFT');
		else game.humanMove(dy > 0 ? 'DOWN' : 'UP');
	}
</script>

<svelte:head>
	<title>2048 AI — n-tuple + expectimax in your browser</title>
	<meta
		name="description"
		content="Play 2048 or watch a strong n-tuple + expectimax AI solve it — running entirely client-side, no server."
	/>
</svelte:head>

<svelte:window onkeydown={onKey} />

<main class="mx-auto flex min-h-svh max-w-md flex-col gap-5 px-4 py-6">
	<header class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold tracking-tight">2048 <span class="text-primary">AI</span></h1>
			<p class="text-xs text-muted-foreground">n-tuple + expectimax, 100% in your browser</p>
		</div>
		<Button variant="ghost" size="icon" onclick={toggleMode} aria-label="Toggle theme">
			<Sun class="h-5 w-5 dark:hidden" />
			<Moon class="hidden h-5 w-5 dark:block" />
		</Button>
	</header>

	<div class="grid grid-cols-2 gap-2">
		<div class="rounded-md bg-muted px-3 py-2">
			<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Score</div>
			<div class="text-xl font-bold tabular-nums">{game.score}</div>
		</div>
		<div class="rounded-md bg-muted px-3 py-2">
			<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Best</div>
			<div class="text-xl font-bold tabular-nums">{game.best}</div>
		</div>
	</div>

	<div class="flex flex-col gap-1.5">
		<span class="text-xs text-muted-foreground">Board size · one AI plays them all</span>
		<div class="flex flex-wrap gap-1.5">
			{#each SHAPES as s (s.label)}
				<Button
					class="h-8 px-2.5 text-xs"
					variant={game.H === s.H && game.W === s.W ? 'default' : 'outline'}
					onclick={() => game.setShape(s.H, s.W)}
				>
					{s.label}
				</Button>
			{/each}
		</div>
	</div>

	<div role="application" aria-label="2048 board" ontouchstart={onTouchStart} ontouchend={onTouchEnd}>
		<Board sprites={game.sprites} H={game.H} W={game.W} hint={game.hintDir} slideMs={game.slideMs} />
	</div>

	<div class="flex items-center justify-between text-xs text-muted-foreground">
		<Badge variant="secondary">Max&nbsp;{game.maxTile}</Badge>
		<span class="tabular-nums">
			{game.moves} moves{#if game.thinking}&nbsp;· thinking…{/if}
		</span>
		{#if game.status === 'over'}
			<Badge variant="destructive">Game over</Badge>
		{/if}
	</div>

	<Controls {game} />

	{#if game.status === 'over'}
		<div class="rounded-lg border bg-card p-4 text-center text-card-foreground">
			<p class="font-semibold">Game over</p>
			<p class="text-sm text-muted-foreground">
				Max tile {game.maxTile} · Score {game.score} · {game.moves} moves
			</p>
			<Button class="mt-3" onclick={() => game.newGame()}>New game</Button>
		</div>
	{/if}

	<footer class="mt-auto pt-2 text-center text-[11px] text-muted-foreground">
		Arrow keys or swipe to play · press <kbd class="rounded border px-1">r</kbd> to restart
	</footer>
</main>
