/// <reference lib="webworker" />
/**
 * The AI Worker entry for both web hosts.
 *
 * Vite needs the `new Worker(new URL(...))` call in the host to emit a chunk,
 * so the entry lives here and is two lines; the search itself is in
 * `@2048/core`, shared with the Kunkun plugin page.
 */
import { startAiWorker, type WorkerScope } from "@2048/core";

startAiWorker(self as unknown as WorkerScope);
