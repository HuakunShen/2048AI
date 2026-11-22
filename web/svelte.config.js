import adapter from '@sveltejs/adapter-cloudflare';

/**
 * Svelte and kit options in one place, so `svelte-check` sees exactly what the
 * build uses. (Options passed inline to `sveltekit()` in `vite.config.ts` make
 * this file ignored, and `svelte-check` then reports the custom-element warning.)
 *
 * `customElement` is a filename predicate: only `@2048/ui`'s components compile
 * to a custom element; the app's own components stay ordinary components.
 */
export default {
	compilerOptions: {
		// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
		runes: ({ filename }) =>
			filename.split(/[/\\]/).includes('node_modules') ? undefined : true,
		customElement: ({ filename }) => filename.includes('/packages/ui/'),
	},
	kit: { adapter: adapter() },
};
