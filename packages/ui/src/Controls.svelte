<script lang="ts">
	import {
		DEFAULT_SPEED_MS,
		DELAY_MAX,
		DELAY_MIN,
		STRENGTHS,
		type GameSnapshot
	} from "@2048/core";

	let {
		snapshot,
		onNew,
		onToggleAuto,
		onStep,
		onHint,
		onLevel,
		onSpeed
	}: {
		snapshot: GameSnapshot;
		onNew: () => void;
		onToggleAuto: () => void;
		onStep: () => void;
		onHint: () => void;
		onLevel: (level: number) => void;
		onSpeed: (speedMs: number) => void;
	} = $props();

	const playable = $derived(snapshot.aiReady && snapshot.status === "playing");

	// The engine wants a *delay* (ms/move), but a "speed" slider must run
	// slow→fast left→right. So the slider tracks speed and we mirror it onto the
	// delay: the right end (DELAY_MAX) maps to the smallest delay (DELAY_MIN).
	const toSpeed = (delay: number) => DELAY_MIN + DELAY_MAX - delay;
	const sliderValue = $derived(toSpeed(snapshot.speedMs));
</script>

<div class="controls">
	{#if !snapshot.aiReady && snapshot.aiError === null && snapshot.status === "playing"}
		<div class="progress-block">
			<div class="row muted small">
				<span>Loading AI model…</span>
				<span class="num">{Math.round(snapshot.progress * 100)}%</span>
			</div>
			<div class="progress"><div class="progress-fill" style="width: {snapshot.progress * 100}%"></div></div>
		</div>
	{:else if snapshot.aiError !== null}
		<p class="muted small ai-error" title={snapshot.aiError}>
			AI unavailable — play with the arrow keys. {snapshot.aiError}
		</p>
	{/if}

	<div class="grid-4">
		<button class="btn" type="button" onclick={onNew}>New</button>
		<button class="btn primary" type="button" onclick={onToggleAuto} disabled={!playable}>
			{snapshot.auto ? "Pause" : "Auto"}
		</button>
		<button
			class="btn outline"
			type="button"
			onclick={onStep}
			disabled={!playable || snapshot.auto || snapshot.thinking}>Step</button
		>
		<button
			class="btn outline"
			type="button"
			onclick={onHint}
			disabled={!playable || snapshot.auto || snapshot.thinking}>Hint</button
		>
	</div>

	<div class="field">
		<div class="row muted small">
			<span>AI strength</span>
			<span class="tiny">{STRENGTHS.find((s) => s.level === snapshot.level)?.hint}</span>
		</div>
		<div class="grid-3">
			{#each STRENGTHS as s (s.level)}
				<button
					class="btn"
					class:primary={snapshot.level === s.level}
					class:outline={snapshot.level !== s.level}
					type="button"
					title={s.hint}
					onclick={() => onLevel(s.level)}>{s.label}</button
				>
			{/each}
		</div>
	</div>

	<div class="field">
		<div class="row muted small">
			<span>Auto-play speed</span>
			<span class="num">{snapshot.speedMs} ms/move</span>
		</div>
		<input
			class="slider"
			type="range"
			min={DELAY_MIN}
			max={DELAY_MAX}
			step={20}
			value={sliderValue}
			oninput={(event) => onSpeed(toSpeed(Number((event.currentTarget as HTMLInputElement).value)))}
		/>
	</div>
</div>

<style>
	.controls {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		font-size: var(--font-size-base, 1rem);
	}
	.row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
	}
	.muted {
		color: var(--muted-foreground, GrayText);
	}
	.small {
		font-size: 0.75rem;
	}
	.tiny {
		font-size: 0.6875rem;
	}
	.num {
		font-variant-numeric: tabular-nums;
	}
	.progress-block,
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.progress {
		height: 0.5rem;
		border-radius: 9999px;
		background: var(--muted, rgb(127 127 127 / 0.25));
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: var(--primary, CanvasText);
		transition: width 120ms linear;
	}
	.grid-4 {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.375rem;
	}
	.grid-3 {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 0.375rem;
	}
	.btn {
		height: 2rem;
		padding: 0 0.5rem;
		border-radius: var(--radius, 6px);
		border: 1px solid var(--border, GrayText);
		background: var(--secondary, transparent);
		color: var(--secondary-foreground, CanvasText);
		font: inherit;
		font-weight: 500;
		font-size: 0.8125rem;
		cursor: pointer;
		transition: opacity 120ms ease;
	}
	.btn:hover:not(:disabled) {
		opacity: 0.8;
	}
	.btn:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.btn:focus-visible {
		outline: 2px solid var(--ring, Highlight);
		outline-offset: 2px;
	}
	.btn.primary {
		border-color: transparent;
		background: var(--primary, CanvasText);
		color: var(--primary-foreground, Canvas);
	}
	.btn.outline {
		background: transparent;
		color: var(--foreground, CanvasText);
	}
	.slider {
		width: 100%;
		accent-color: var(--primary, CanvasText);
	}
	.ai-error {
		margin: 0;
		overflow-wrap: anywhere;
	}
</style>
