<script lang="ts">
	import '../app.css';
	import { ModeWatcher, Toaster } from '@kksh/svelte5';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<ModeWatcher />
<Toaster richColors position="top-center" />

{@render children()}
