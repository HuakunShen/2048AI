/**
 * Where the best score and the last board shape are kept.
 *
 * An interface rather than a direct `localStorage` read, so the core stays
 * DOM-free and the tests stay deterministic. A host that has no persistence
 * passes nothing and gets {@link memoryStorage}, which behaves exactly like the
 * original web app (best score lasted for the session only).
 */
export interface GameStorage {
	get(key: string): string | null;
	set(key: string, value: string): void;
}

export const BEST_KEY = "game-2048:best";
export const SHAPE_KEY = "game-2048:shape";

/** A process-lifetime store. Used when a host does not pass one. */
export function memoryStorage(): GameStorage {
	const values = new Map<string, string>();
	return {
		get: (key) => values.get(key) ?? null,
		set: (key, value) => {
			values.set(key, value);
		},
	};
}

/**
 * A store backed by a Web Storage area. Reads and writes are guarded: private
 * browsing and disabled storage throw on access, and a game must not break
 * because a best score cannot be saved.
 */
export function webStorage(area: Pick<Storage, "getItem" | "setItem">): GameStorage {
	return {
		get(key) {
			try {
				return area.getItem(key);
			} catch {
				return null;
			}
		},
		set(key, value) {
			try {
				area.setItem(key, value);
			} catch {
				// A game does not fail because storage is unavailable.
			}
		},
	};
}
