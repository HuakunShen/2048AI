/**
 * Tile visuals — the classic 2048 palette, extended past 2048.
 *
 * The board's colours are deliberately fixed rather than derived from the shell
 * accent, so a game looks like 2048 everywhere. The surrounding chrome reads the
 * plugin theme tokens instead (see the component styles).
 */
const TILE_BG: Record<number, string> = {
	0: "#cdc1b4",
	2: "#eee4da",
	4: "#ede0c8",
	8: "#f2b179",
	16: "#f59563",
	32: "#f67c5f",
	64: "#f65e3b",
	128: "#edcf72",
	256: "#edcc61",
	512: "#edc850",
	1024: "#edc53f",
	2048: "#edc22e",
	4096: "#3c3a32",
	8192: "#2e6f6a",
	16384: "#1f7a5a",
	32768: "#1565c0",
};

export function tileBg(value: number): string {
	return TILE_BG[value] ?? "#0f4c81";
}

export function tileFg(value: number): string {
	return value <= 4 ? "#776e65" : "#f9f6f2";
}

/** A size class instead of Tailwind utilities, since the element carries its own CSS. */
export function tileSizeClass(value: number): string {
	if (value < 100) return "size-lg";
	if (value < 1000) return "size-md";
	if (value < 10000) return "size-sm";
	return "size-xs";
}
