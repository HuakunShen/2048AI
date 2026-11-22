/// <reference lib="webworker" />
/**
 * The plugin page's AI Worker entry. Vite needs the `new Worker(new URL(...))`
 * call in the page to emit a chunk, so the entry lives here; the search is the
 * same `@2048/core` code the web app uses.
 */
import { startAiWorker, type WorkerScope } from "@2048/core";

startAiWorker(self as unknown as WorkerScope);
