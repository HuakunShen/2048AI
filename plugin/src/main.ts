/**
 * The frame page.
 *
 * Host-agnostic on purpose: it mounts `<game-2048>`, starts the AI Worker and
 * attaches the model, then exposes the element for a host's own glue module.
 * Kunkun's `connectHost()` lives in the extension that stages this page, so this
 * repository never depends on the (unpublished) Kunkun SDK.
 */
import "@2048/ui";
import { createWorkerAi } from "@2048/core";

const element = document.getElementById("game");
if (element === null) throw new Error("index.html has no #game element");

const modelUrl = new URL("model.pack", document.baseURI).href;
/** A new Worker per attempt, so Retry after a dead Worker starts clean. */
const makeAi = () => createWorkerAi(new Worker(new URL("./ai-worker.ts", import.meta.url), { type: "module" }));

await customElements.whenDefined("game-2048");

const game = element as HTMLElement & {
	attachAi(provider: unknown, url: string): Promise<void>;
	newGame(): void;
};

void game.attachAi(makeAi, modelUrl);

declare global {
	interface Window {
		/** The mounted element, for a host's glue module (Kunkun's theme, AI tools). */
		__game2048?: HTMLElement;
	}
}
window.__game2048 = element;
