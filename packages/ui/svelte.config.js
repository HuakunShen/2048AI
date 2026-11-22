/**
 * Only the root component is a custom element; `Board`, `Tile` and `Controls`
 * are ordinary components imported by it.
 */
export default {
	compilerOptions: {
		customElement: ({ filename }) => filename.endsWith('Game.svelte'),
	},
};
