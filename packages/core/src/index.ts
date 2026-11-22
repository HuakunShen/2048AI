export { DIRS, Engine, MAX_EXPONENT, expToTile, tileToExp, tilesToBoard, boardToTiles } from "./engine";
export type { Board, Dir, MovePlan, MoveResult, Rng, Slide } from "./engine";

export { Expectimax } from "./ai/expectimax";
export type { DepthCfg, ShapeValueFn } from "./ai/expectimax";
export { UniversalValue, scatterInto, tableSize } from "./ai/universal";
export { compilePattern, coreLibrary, orientCells } from "./ai/patterns";
export type { CompiledPattern, Pattern } from "./ai/patterns";
export { buildValueFromPacked, patternsFromManifest } from "./ai/model";
export type { PackedManifest } from "./ai/model";
export { startAiWorker } from "./ai/worker";
export type { WorkerScope } from "./ai/worker";
export { AiUnavailableError, createWorkerAi } from "./ai/client";
export type { AiErrorCode, WorkerRequest, WorkerResponse } from "./ai/protocol";

export {
	GameController,
	SHAPES,
	STRENGTHS,
	DELAY_MAX,
	DELAY_MIN,
	DEFAULT_SPEED_MS,
	SLIDE_MS,
	depthCfgFor,
} from "./game/controller";
export type {
	AiMove,
	AiProvider,
	GameOptions,
	GameSnapshot,
	Scheduler,
	Status,
} from "./game/controller";
export { finalizeSprites, slideSprites, spritesFromBoard } from "./game/sprites";
export type { Sprite } from "./game/sprites";
export { BEST_KEY, SHAPE_KEY, memoryStorage, webStorage } from "./game/storage";
export type { GameStorage } from "./game/storage";
