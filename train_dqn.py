"""Headless Double-DQN trainer for 2048.

Runs entirely without a GUI. The game simulation uses the fast numba-jit
``NumpyStaticBoard`` on CPU; only the neural network runs on the accelerator
(Apple MPS / CUDA / CPU fallback). Progress is logged to TensorBoard under
``runs/`` and checkpoints are written under ``models/``.

Examples
--------
    uv run train_dqn.py                        # sensible defaults
    uv run train_dqn.py --episodes 3000        # train longer
    uv run train_dqn.py --smoke                # 5-episode sanity run
"""
import time
import argparse
from pathlib import Path

import torch

from src.game.controller.game import Game
from src.game.model.staticboardImpl import NumpyStaticBoard
from src.agent.agentImpl import DQNPlayer


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser("2048 DQN headless trainer")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=80000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--empty-weight", type=float, default=0.5,
                        help="Potential-based empty-cell reward shaping weight (0 disables)")
    parser.add_argument("--reward-clip", type=float, default=1.0,
                        help="Clip per-step reward to [-c, c] to keep Q bounded (0 disables)")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the env seed (default: None => diverse episodes)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run to sanity-check the pipeline")
    parser.add_argument("--resume", default=None,
                        help="Path to a .pth checkpoint to resume training from "
                             "(restores weights, optimizer, epsilon, total_steps)")
    args = parser.parse_args()

    if args.smoke:
        args.episodes = 5
        args.eval_every = 5
        args.eval_episodes = 5
        args.save_every = 0
        args.epsilon_decay_steps = 2000

    device = get_device(args.device)
    print(f"Using device: {device}")

    timestamp = int(time.time())
    model_name = f"dqn_2048_{timestamp}"
    checkpoint_dir = Path("models") / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = f"runs/{model_name}"
    print(f"Checkpoints -> {checkpoint_dir}\nTensorBoard -> {log_dir}")

    # Headless game: numba CPU backend, seed=None => each episode differs.
    game = Game(seed=args.seed, static_board=NumpyStaticBoard)

    agent = DQNPlayer(
        game=game,
        quiet=False,
        ui=False,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_min=args.epsilon_min,
        epsilon_decay_steps=args.epsilon_decay_steps,
        learning_rate=args.lr,
        target_update=args.target_update,
        reward_scale=args.reward_scale,
        train_every=args.train_every,
        empty_weight=args.empty_weight,
        reward_clip=args.reward_clip,
        device=device,
    )

    if args.resume:
        agent.load_model(args.resume)
        print(f"Resumed from {args.resume} at {agent.total_steps} steps, epsilon={agent.epsilon:.3f}")

    print(f"Starting training for {args.episodes} episodes...")
    t0 = time.time()
    try:
        agent.train(
            episodes=args.episodes,
            log_dir=log_dir,
            checkpoint_dir=str(checkpoint_dir),
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
        )
    except KeyboardInterrupt:
        print("\nInterrupted - saving current model...")
        agent.save_model(str(checkpoint_dir / "interrupted_model.pth"))

    agent.save_model(str(checkpoint_dir / "final_model.pth"))
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed / 60:.1f} min. Final greedy evaluation...")

    stats = agent.evaluate(episodes=max(50, args.eval_episodes))
    print(f"  mean_score {stats['mean_score']:.0f} | mean_max {stats['mean_max']:.0f} "
          f"| best_max {stats['best_max']} | reach2048 {stats['win_rate'] * 100:.0f}%")

    with open(checkpoint_dir / "test_results.txt", "w") as f:
        f.write(
            f"episodes: {args.episodes}\n"
            f"train_minutes: {elapsed / 60:.2f}\n"
            f"mean_score: {stats['mean_score']:.1f}\n"
            f"mean_max_tile: {stats['mean_max']:.1f}\n"
            f"best_max_tile: {stats['best_max']}\n"
            f"reach2048_rate: {stats['win_rate']:.3f}\n"
        )


if __name__ == "__main__":
    main()
