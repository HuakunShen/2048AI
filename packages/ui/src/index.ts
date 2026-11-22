/**
 * `@2048/ui` — the game as one custom element, shared by every host.
 *
 * Importing this module defines `<game-2048>`. It has no framework peer: the
 * Svelte implementation is compiled to a custom element, so a React page, a
 * plain HTML page and the SvelteKit app all mount the same code.
 */
import "./Game.svelte";

export type { Dir, GameSnapshot } from "@2048/core";
