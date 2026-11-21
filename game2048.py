"""A deterministic, seedable 2048 environment for reinforcement learning."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


class Game2048:
    """Classic 4x4 2048 rules with a Gym-like reset/step interface.

    Observations are float32 arrays shaped (1, 4, 4). Each occupied cell is
    encoded as log2(tile) / log2(target_tile), so the target tile is 1.0.
    Actions are integers: 0=up, 1=right, 2=down, 3=left.
    """

    ACTIONS = ("up", "right", "down", "left")
    SIZE = 4

    def __init__(
        self,
        seed: int | None = None,
        target_tile: int = 2048,
        continue_after_win: bool = False,
    ) -> None:
        if target_tile < 4 or target_tile & (target_tile - 1):
            raise ValueError("target_tile must be a power of two of at least 4")

        self.target_tile = target_tile
        self.continue_after_win = continue_after_win
        self._target_exponent = math.log2(target_tile)
        self._rng = random.Random(seed)
        self.board: list[list[int]] = []
        self.score = 0
        self.won = False
        self.game_over = False
        self.done = False
        self.moves = 0
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh game and return the initial observation and info."""
        if seed is not None:
            self._rng.seed(seed)

        self.board = [[0] * self.SIZE for _ in range(self.SIZE)]
        self.score = 0
        self.won = False
        self.game_over = False
        self.done = False
        self.moves = 0
        self._spawn_tile()
        self._spawn_tile()
        self.done = not self.valid_actions()
        return self.observation(), self._info(score_delta=0)

    def observation(self, max_exponent: float | None = None) -> np.ndarray:
        """Return normalized tile exponents; override scale for N-tuple play."""
        scale = self._target_exponent if max_exponent is None else float(max_exponent)
        if scale <= 0:
            raise ValueError("max_exponent must be positive")
        encoded = np.zeros((1, self.SIZE, self.SIZE), dtype=np.float32)
        for row in range(self.SIZE):
            for column in range(self.SIZE):
                tile = self.board[row][column]
                if tile:
                    encoded[0, row, column] = min(
                        1.0, math.log2(tile) / scale
                    )
        return encoded

    def valid_actions(self) -> tuple[int, ...]:
        """Return actions that change the current board, in fixed action order."""
        valid: list[int] = []
        for action, direction in enumerate(self.ACTIONS):
            moved_board, _ = self._apply_move(direction)
            if moved_board != self.board:
                valid.append(action)
        return tuple(valid)

    def step(
        self, action: int | str
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply one legal move and return observation, reward, done, info.

        Reward is log-scaled merge score, a small survival bonus, a win bonus,
        and a loss penalty. Invalid moves raise ValueError and never spawn tiles.
        """
        if self.done:
            raise RuntimeError("step() called after the episode ended; call reset()")

        action_index = self._coerce_action(action)
        if action_index not in self.valid_actions():
            raise ValueError(f"illegal move: {self.ACTIONS[action_index]}")

        direction = self.ACTIONS[action_index]
        moved_board, score_delta = self._apply_move(direction)
        self.board = moved_board
        self.score += score_delta
        self.moves += 1

        reward = math.log2(1.0 + score_delta) if score_delta else 0.0
        reward += 0.02
        reached_target = any(
            self.board[row][column] >= self.target_tile
            for row in range(self.SIZE)
            for column in range(self.SIZE)
        )
        just_won = reached_target and not self.won
        self.won = self.won or reached_target

        # The original game spawns a tile after every successful move,
        # including the move that creates 2048.
        self._spawn_tile()
        self.game_over = not self.valid_actions()
        self.done = self.game_over or (self.won and not self.continue_after_win)
        if just_won:
            reward += 25.0
        if self.game_over and not self.won:
            reward -= 2.0

        return self.observation(), reward, self.done, self._info(score_delta)

    def render(self) -> str:
        """Return a readable text board for training logs and debugging."""
        cell_width = max(4, len(str(max(self.target_tile, self._max_tile()))))
        divider = "+" + "+".join("-" * (cell_width + 2) for _ in range(self.SIZE)) + "+"
        rows = [divider]
        for row in self.board:
            rows.append(
                "|"
                + "|".join(
                    f"{(str(value) if value else '.'):>{cell_width + 2}}" for value in row
                )
                + "|"
            )
            rows.append(divider)
        return "\n".join(rows)

    def _coerce_action(self, action: int | str) -> int:
        if isinstance(action, str):
            try:
                return self.ACTIONS.index(action.lower())
            except ValueError as error:
                raise ValueError(f"unknown direction: {action!r}") from error
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError("action must be an integer action id or direction string")
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError(f"action id must be in [0, {len(self.ACTIONS) - 1}]")
        return action

    def _spawn_tile(self) -> None:
        empty = [
            (row, column)
            for row in range(self.SIZE)
            for column in range(self.SIZE)
            if self.board[row][column] == 0
        ]
        if not empty:
            return
        row, column = self._rng.choice(empty)
        self.board[row][column] = 2 if self._rng.random() < 0.9 else 4

    def _apply_move(self, direction: str) -> tuple[list[list[int]], int]:
        if direction not in self.ACTIONS:
            raise ValueError(f"unknown direction: {direction!r}")

        result = [[0] * self.SIZE for _ in range(self.SIZE)]
        score_delta = 0

        for line_index in range(self.SIZE):
            coordinates = self._line_coordinates(direction, line_index)
            compacted = [
                self.board[row][column]
                for row, column in coordinates
                if self.board[row][column] != 0
            ]
            merged: list[int] = []
            value_index = 0
            while value_index < len(compacted):
                if (
                    value_index + 1 < len(compacted)
                    and compacted[value_index] == compacted[value_index + 1]
                ):
                    new_value = compacted[value_index] * 2
                    merged.append(new_value)
                    score_delta += new_value
                    value_index += 2
                else:
                    merged.append(compacted[value_index])
                    value_index += 1

            for offset, (row, column) in enumerate(coordinates):
                if offset < len(merged):
                    result[row][column] = merged[offset]

        return result, score_delta

    def _line_coordinates(self, direction: str, line_index: int) -> list[tuple[int, int]]:
        if direction == "left":
            return [(line_index, column) for column in range(self.SIZE)]
        if direction == "right":
            return [(line_index, column) for column in range(self.SIZE - 1, -1, -1)]
        if direction == "up":
            return [(row, line_index) for row in range(self.SIZE)]
        return [(row, line_index) for row in range(self.SIZE - 1, -1, -1)]

    def _max_tile(self) -> int:
        return max((value for row in self.board for value in row), default=0)

    def _info(self, score_delta: int) -> dict[str, Any]:
        return {
            "score": self.score,
            "score_delta": score_delta,
            "max_tile": self._max_tile(),
            "won": self.won,
            "game_over": self.game_over,
            "moves": self.moves,
        }
