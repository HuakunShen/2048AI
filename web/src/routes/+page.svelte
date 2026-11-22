<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { Button } from '@kksh/svelte5';
	import { Moon, Sun } from 'lucide-svelte';
	import { toggleMode } from 'mode-watcher';
	import '@2048/ui';
	import { createWorkerAi } from '@2048/core';

	// The game itself is the shared `<game-2048>` custom element (`@2048/ui`), the
	// same one the Kunkun plugin page mounts. This page only adds the web shell:
	// the title, the theme toggle and the footer.
	type GameElement = HTMLElement & { attachAi(provider: unknown, modelUrl: string): Promise<void> };

	let element = $state<GameElement | undefined>(undefined);
	let announced = false;

	function onScore(event: Event): void {
		const detail = (event as CustomEvent<{ maxTile: number }>).detail;
		if (announced || detail.maxTile < 2048) return;
		announced = true;
		toast.success('Reached 2048! 🎉', { description: 'Keep going — the AI can hit 4096+.' });
	}

	function onNewGame(): void {
		announced = false;
	}

	onMount(() => {
		let cancelled = false;
		let dispose: (() => void) | undefined;

		void (async () => {
			await customElements.whenDefined('game-2048');
			if (cancelled || element === undefined) return;
			const worker = new Worker(new URL('../lib/ai-worker.ts', import.meta.url), { type: 'module' });
			const ai = createWorkerAi(worker);
			dispose = () => ai.dispose();
			await element.attachAi(ai, new URL('model/parts.json', document.baseURI).href);
		})();

		return () => {
			cancelled = true;
			dispose?.();
		};
	});
</script>

<svelte:head>
	<title>2048 AI — n-tuple + expectimax in your browser</title>
	<meta
		name="description"
		content="Play 2048 or watch a strong n-tuple + expectimax AI solve it — running entirely client-side, no server."
	/>
</svelte:head>

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

	<game-2048 bind:this={element} onscore={onScore} ongameover={onNewGame}></game-2048>

	<footer class="mt-auto text-center text-[11px] text-muted-foreground">
		One engine, one AI, any board size — shared with the Kunkun plugin.
	</footer>
</main>
