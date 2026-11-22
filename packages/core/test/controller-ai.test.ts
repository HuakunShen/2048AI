import { describe, expect, it } from "vitest";
import { GameController, type AiProvider, type AiMove, type Scheduler } from "../src/game/controller";

/** A provider whose behaviour each test controls; records what the controller asked for. */
function fakeProvider(behaviour: {
	init?: () => Promise<void>;
	move?: (call: number) => Promise<AiMove>;
}) {
	const calls: number[] = [];
	let disposed = 0;
	const provider: AiProvider = {
		init: behaviour.init ?? (async () => {}),
		bestMove: async () => {
			calls.push(calls.length + 1);
			return behaviour.move ? behaviour.move(calls.length) : { dir: "LEFT", value: 1 };
		},
		dispose: () => {
			disposed += 1;
		},
	};
	return { provider, calls, disposed: () => disposed };
}

/** A scheduler that records timers instead of running them, so a test decides when a loop proceeds. */
function manualScheduler() {
	const timers: { ms: number; fn: () => void; cancelled: boolean }[] = [];
	const scheduler: Scheduler = {
		after(ms, fn) {
			const timer = { ms, fn, cancelled: false };
			timers.push(timer);
			return () => {
				timer.cancelled = true;
			};
		},
	};
	const run = () => {
		for (const timer of timers.splice(0)) if (!timer.cancelled) timer.fn();
	};
	return { scheduler, timers, run };
}

const tick = (ms = 5) => new Promise((resolve) => setTimeout(resolve, ms));

describe("controller AI failure paths", () => {
	it("Retry builds a fresh provider, disposes the failed one and reloads", async () => {
		const failed = fakeProvider({ init: async () => Promise.reject(new Error("model fetch failed")) });
		const healthy = fakeProvider({});
		let built = 0;
		const controller = new GameController({ H: 4, W: 4 });

		await controller.boot(() => (++built === 1 ? failed.provider : healthy.provider), "model.pack");
		expect(controller.getSnapshot().aiError).toContain("model fetch failed");
		expect(controller.getSnapshot().aiReady).toBe(false);

		await controller.retry();
		const snapshot = controller.getSnapshot();
		expect(built).toBe(2);
		expect(failed.disposed()).toBe(1);
		expect(snapshot.aiError).toBeNull();
		expect(snapshot.aiReady).toBe(true);
		expect(snapshot.progress).toBe(1);
	});

	it("a rejected search stops auto-play and leaves the board playable", async () => {
		const ai = fakeProvider({ move: async () => Promise.reject(new Error("TIMEOUT")) });
		const controller = new GameController({ H: 4, W: 4 });
		await controller.boot(ai.provider, "model.pack");
		expect(controller.getSnapshot().aiReady).toBe(true);

		controller.toggleAuto();
		await tick(20);
		const snapshot = controller.getSnapshot();
		expect(snapshot.auto).toBe(false);
		expect(snapshot.thinking).toBe(false);
		expect(snapshot.aiError).toContain("TIMEOUT");

		// The player can still move: `auto` no longer swallows the move.
		const moved = ["LEFT", "UP", "RIGHT", "DOWN"].some((dir) => controller.humanMove(dir as never));
		expect(moved).toBe(true);
	});

	it("pause during an in-flight move does not leave a second auto loop running", async () => {
		const pending: ((move: AiMove) => void)[] = [];
		const ai = fakeProvider({
			move: () =>
				new Promise<AiMove>((resolve) => {
					pending.push(resolve);
				}),
		});
		const sched = manualScheduler();
		/** Only the auto-play delay (speedMs), not the slide animation (slideMs). */
		const autoTimers = () => sched.timers.filter((timer) => timer.ms === 140);
		const controller = new GameController({ H: 4, W: 4, scheduler: sched.scheduler });
		await controller.boot(ai.provider, "model.pack");

		controller.toggleAuto();
		await tick();
		expect(ai.calls).toHaveLength(1); // loop A is inside its first search

		controller.toggleAuto(); // Pause while the search is still pending
		expect(controller.getSnapshot().auto).toBe(false);

		controller.toggleAuto(); // Auto again: loop B starts its own search
		await tick();
		expect(ai.calls).toHaveLength(2);

		// The move loop A was waiting for arrives late. It must not re-arm a timer.
		expect(pending).toHaveLength(2);
		pending[0]?.({ dir: "LEFT", value: 1 });
		await tick();
		expect(autoTimers()).toHaveLength(0);
		expect(controller.getSnapshot().auto).toBe(true); // loop B still owns auto

		pending[1]?.({ dir: "LEFT", value: 1 });
		await tick();
		expect(autoTimers()).toHaveLength(1); // only loop B scheduled the next move
	});
});
