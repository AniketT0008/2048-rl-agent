"""Evaluate a saved N-tuple checkpoint on fresh seeded games."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ntuple_agent import TABLE_SHAPE, evaluate_games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=500_000)
    parser.add_argument("--search-depth", choices=("1", "2", "both"), default="both")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.games < 1:
        parser.error("--games must be positive")

    loaded = np.load(args.checkpoint, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            weights = np.array(loaded["weights"], dtype=np.float32, copy=True)
    else:
        weights = np.asarray(loaded, dtype=np.float32)
    if weights.shape != TABLE_SHAPE:
        parser.error(f"checkpoint weights have shape {weights.shape}, expected {TABLE_SHAPE}")

    search_depths = (1, 2) if args.search_depth == "both" else (int(args.search_depth),)
    for search_depth in search_depths:
        results = evaluate_games(
            weights,
            args.games,
            args.seed,
            search_depth=search_depth,
        )
        print(
            f"games={args.games} search_depth={search_depth} "
            f"win_rate={results['win_rate']:.1%} "
            f"mean_score={results['mean_score']:.0f} "
            f"median_score={results['median_score']:.0f} "
            f"mean_max_tile={results['mean_max_tile']:.0f} "
            f"best_tile={results['best_tile']} "
            f"best_tile_score={results['best_tile_score']} "
            f"best_seed={results['best_tile_seed']} "
            f"max_score={results['max_score']}"
        )


if __name__ == "__main__":
    main()
