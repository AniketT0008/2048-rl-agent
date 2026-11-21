"""Train N-tuple TD-afterstate value weights for the 2048 game."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ntuple_agent import (
    FEATURE_CELLS,
    TABLE_SHAPE,
    create_empty_weights,
    evaluate_games,
    seed_numba,
    train_one_episode,
)


def save_weights(path: Path, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as checkpoint_file:
        np.savez_compressed(checkpoint_file, weights=weights)


def load_weights(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            return np.array(loaded["weights"], dtype=np.float32, copy=True)
    return np.array(loaded, dtype=np.float32, copy=True)


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "ntuple_best.npz"
    metrics_path = output_dir / "ntuple_training.csv"

    if args.resume is not None:
        loaded = load_weights(args.resume)
        if loaded.shape != TABLE_SHAPE or loaded.dtype != np.float32:
            raise ValueError(f"resume weights must be float32 with shape {TABLE_SHAPE}")
        weights = loaded
    else:
        weights = create_empty_weights()

    seed_numba(args.seed)
    best_win_rate = -1.0
    best_mean_score = -1.0
    best_max_score = -1
    started = time.perf_counter()
    deadline = (
        started + args.duration_minutes * 60
        if args.duration_minutes is not None
        else None
    )

    if args.resume is not None:
        baseline = evaluate_games(
            weights,
            args.eval_games,
            args.eval_seed,
            search_depth=args.search_depth,
        )
        best_win_rate = baseline["win_rate"]
        best_mean_score = baseline["mean_score"]
        best_max_score = baseline["max_score"]
        save_weights(best_path, weights)
        print(
            f"resume baseline | win rate {best_win_rate:.1%} | "
            f"mean score {best_mean_score:,.0f} | max score {best_max_score:,} | "
            f"best tile {baseline['best_tile']:,}"
        )

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=("episode", "epsilon", "score", "max_tile", "won", "moves"),
        )
        writer.writeheader()

        completed_episodes = 0
        for episode in range(args.episodes):
            if deadline is not None and time.perf_counter() >= deadline:
                print(f"Training time limit reached after {episode:,} episodes.")
                break
            fraction = episode / max(args.episodes - 1, 1)
            epsilon = args.epsilon_start + fraction * (
                args.epsilon_end - args.epsilon_start
            )
            score, max_tile, won, moves = train_one_episode(
                weights, epsilon, args.learning_rate, args.max_steps
            )
            completed_episodes = episode + 1
            writer.writerow(
                {
                    "episode": episode + 1,
                    "epsilon": f"{epsilon:.5f}",
                    "score": score,
                    "max_tile": max_tile,
                    "won": int(won),
                    "moves": moves,
                }
            )
            if (episode + 1) % args.log_every == 0:
                metrics_file.flush()
                elapsed = time.perf_counter() - started
                print(
                    f"episode {episode + 1:,}/{args.episodes:,} | "
                    f"epsilon {epsilon:.3f} | "
                    f"latest score {score:,} | max tile {max_tile:,} | "
                    f"episode rate {(episode + 1) / max(elapsed, 1e-9):.0f}/s"
                )

            if (
                (episode + 1) % args.eval_every == 0
                or episode + 1 == args.episodes
                or (deadline is not None and time.perf_counter() >= deadline)
            ):
                metrics_file.flush()
                results = evaluate_games(
                    weights,
                    args.eval_games,
                    args.eval_seed + episode * 10,
                    search_depth=args.search_depth,
                )
                print(
                    f"evaluation at {episode + 1:,} episodes | "
                    f"win rate {results['win_rate']:.1%} | "
                    f"mean score {results['mean_score']:,.0f} | "
                    f"median score {results['median_score']:,.0f} | "
                    f"mean max tile {results['mean_max_tile']:,.0f} | "
                    f"best tile {results['best_tile']:,} "
                    f"(score {results['best_tile_score']:,}) | "
                    f"max score {results['max_score']:,} | "
                    f"best win rate {max(best_win_rate, results['win_rate']):.1%}"
                )
                if args.objective == "score":
                    is_better = (
                        results["mean_score"] > best_mean_score
                        or (
                            results["mean_score"] == best_mean_score
                            and results["max_score"] > best_max_score
                        )
                    )
                else:
                    is_better = (
                        results["win_rate"] > best_win_rate
                        or (
                            results["win_rate"] == best_win_rate
                            and results["mean_score"] > best_mean_score
                        )
                    )
                if is_better:
                    best_win_rate = results["win_rate"]
                    best_mean_score = results["mean_score"]
                    best_max_score = results["max_score"]
                    save_weights(best_path, weights)
                if (
                    args.objective == "win_rate"
                    and results["win_rate"] >= args.target_win_rate
                ):
                    print("Requested evaluation win-rate target reached.")
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    print("Time limit reached; final score evaluation complete.")
                    break

    elapsed = time.perf_counter() - started
    print(f"training elapsed: {elapsed / 60:.1f} minutes")
    print(f"best checkpoint: {best_path.resolve()}")
    print(f"training metrics: {metrics_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=700_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=0.0)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument("--target-win-rate", type=float, default=1.0)
    parser.add_argument("--objective", choices=("win_rate", "score"), default="win_rate")
    parser.add_argument("--search-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--output-dir", default="runs_ntuple")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_games < 1:
        parser.error("--episodes and --eval-games must be positive")
    if args.eval_every < 1 or args.log_every < 1 or args.max_steps < 1:
        parser.error("evaluation interval, log interval, and max steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.duration_minutes is not None and args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    if not 0.0 <= args.epsilon_start <= 1.0 or not 0.0 <= args.epsilon_end <= 1.0:
        parser.error("epsilon values must be between 0 and 1")
    if not 0.0 <= args.target_win_rate <= 1.0:
        parser.error("--target-win-rate must be between 0 and 1")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"resume checkpoint not found: {args.resume}")
    return args


if __name__ == "__main__":
    train(parse_args())
