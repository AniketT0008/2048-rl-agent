"""N-tuple temporal-difference RL for 2048, with Numba-compiled hot paths."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numba import njit


WIN_EXPONENT = 11  # 2**11 == 2048
MAX_TUPLE_EXPONENT = 15  # Four-bit cells cap at 2**15 for tuple indexing.
ACTION_COUNT = 4
PATTERN_COUNT = 4
SYMMETRY_COUNT = 8
FEATURE_COUNT = PATTERN_COUNT * SYMMETRY_COUNT
PATTERN_SIZE = 6
TABLE_SIZE = 1 << (4 * PATTERN_SIZE)
TABLE_SHAPE = (PATTERN_COUNT, TABLE_SIZE)


def _build_feature_cells() -> np.ndarray:
    patterns = (
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
        ((1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
    )
    features: list[list[int]] = []
    for pattern in patterns:
        for horizontal_flip in (False, True):
            for quarter_turns in range(4):
                cells: list[int] = []
                for original_row, original_column in pattern:
                    row = original_row
                    column = 3 - original_column if horizontal_flip else original_column
                    for _ in range(quarter_turns):
                        row, column = column, 3 - row
                    cells.append(row * 4 + column)
                features.append(cells)
    return np.asarray(features, dtype=np.int64)


FEATURE_CELLS = _build_feature_cells()


@njit(cache=True)
def seed_numba(seed: int) -> None:
    np.random.seed(seed % 2_147_483_647)


@njit(cache=True)
def _spawn_tile(board: np.ndarray) -> bool:
    empty_count = 0
    for cell in range(16):
        if board[cell] == 0:
            empty_count += 1
    if empty_count == 0:
        return False

    selected_empty = np.random.randint(0, empty_count)
    for cell in range(16):
        if board[cell] == 0:
            if selected_empty == 0:
                board[cell] = 1 if np.random.random() < 0.9 else 2
                return True
            selected_empty -= 1
    return False


@njit(cache=True)
def _new_board() -> np.ndarray:
    board = np.zeros(16, dtype=np.uint8)
    _spawn_tile(board)
    _spawn_tile(board)
    return board


@njit(cache=True)
def _apply_move(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Return the deterministic afterstate, merge score, and changed flag."""
    result = np.zeros(16, dtype=np.uint8)
    score_delta = 0

    for line in range(4):
        positions = np.empty(4, dtype=np.int64)
        values = np.empty(4, dtype=np.uint8)
        value_count = 0

        for offset in range(4):
            if action == 0:  # up
                cell = offset * 4 + line
            elif action == 1:  # right
                cell = line * 4 + (3 - offset)
            elif action == 2:  # down
                cell = (3 - offset) * 4 + line
            else:  # left
                cell = line * 4 + offset
            positions[offset] = cell
            exponent = board[cell]
            if exponent != 0:
                values[value_count] = exponent
                value_count += 1

        merged = np.empty(4, dtype=np.uint8)
        merged_count = 0
        value_index = 0
        while value_index < value_count:
            if (
                value_index + 1 < value_count
                and values[value_index] == values[value_index + 1]
            ):
                new_exponent = int(values[value_index]) + 1
                merged[merged_count] = new_exponent
                score_delta += 1 << new_exponent
                merged_count += 1
                value_index += 2
            else:
                merged[merged_count] = values[value_index]
                merged_count += 1
                value_index += 1

        for offset in range(4):
            if offset < merged_count:
                result[positions[offset]] = merged[offset]

    changed = False
    for cell in range(16):
        if result[cell] != board[cell]:
            changed = True
            break
    return result, score_delta, changed


@njit(cache=True)
def _has_moves(board: np.ndarray) -> bool:
    for action in range(ACTION_COUNT):
        _, _, changed = _apply_move(board, action)
        if changed:
            return True
    return False


@njit(cache=True)
def _feature_indices(board: np.ndarray, feature_cells: np.ndarray) -> np.ndarray:
    indices = np.empty(FEATURE_COUNT, dtype=np.int32)
    for feature in range(FEATURE_COUNT):
        index = 0
        for offset in range(PATTERN_SIZE):
            exponent = min(
                int(board[feature_cells[feature, offset]]), MAX_TUPLE_EXPONENT
            )
            index = (index << 4) | exponent
        indices[feature] = index
    return indices


@njit(cache=True)
def _value_from_indices(indices: np.ndarray, weights: np.ndarray) -> float:
    total = 0.0
    for feature in range(FEATURE_COUNT):
        table = feature // SYMMETRY_COUNT
        total += weights[table, indices[feature]]
    return total


@njit(cache=True)
def _value(board: np.ndarray, weights: np.ndarray, feature_cells: np.ndarray) -> float:
    return _value_from_indices(_feature_indices(board, feature_cells), weights)


@njit(cache=True)
def _update_value(
    board: np.ndarray,
    target: float,
    weights: np.ndarray,
    feature_cells: np.ndarray,
    learning_rate: float,
) -> None:
    indices = _feature_indices(board, feature_cells)
    delta = target - _value_from_indices(indices, weights)
    step = learning_rate * delta / FEATURE_COUNT
    for feature in range(FEATURE_COUNT):
        table = feature // SYMMETRY_COUNT
        weights[table, indices[feature]] += step


@njit(cache=True)
def _choose_action(
    board: np.ndarray,
    weights: np.ndarray,
    feature_cells: np.ndarray,
    epsilon: float,
    legal_mask: np.ndarray,
) -> tuple[int, np.ndarray, int]:
    actions = np.empty(ACTION_COUNT, dtype=np.int8)
    afterstates = np.zeros((ACTION_COUNT, 16), dtype=np.uint8)
    rewards = np.zeros(ACTION_COUNT, dtype=np.int64)
    legal_count = 0

    for action in range(ACTION_COUNT):
        if not legal_mask[action]:
            continue
        afterstate, reward, changed = _apply_move(board, action)
        if changed:
            actions[legal_count] = action
            afterstates[legal_count] = afterstate
            rewards[legal_count] = reward
            legal_count += 1

    if legal_count == 0:
        return -1, np.zeros(16, dtype=np.uint8), 0

    if np.random.random() < epsilon:
        selected = np.random.randint(0, legal_count)
        return int(actions[selected]), afterstates[selected].copy(), int(rewards[selected])

    best_value = -1.0e30
    best_slot = 0
    tie_count = 0
    for slot in range(legal_count):
        afterstate = afterstates[slot]
        future_value = _value(afterstate, weights, feature_cells)
        candidate = float(rewards[slot]) + future_value
        if candidate > best_value + 1.0e-5:
            best_value = candidate
            best_slot = slot
            tie_count = 1
        elif abs(candidate - best_value) <= 1.0e-5:
            tie_count += 1
            if np.random.randint(0, tie_count) == 0:
                best_slot = slot

    return int(actions[best_slot]), afterstates[best_slot].copy(), int(rewards[best_slot])


@njit(cache=True)
def _best_one_ply_value(
    board: np.ndarray,
    weights: np.ndarray,
    feature_cells: np.ndarray,
) -> float:
    best_value = 0.0
    has_action = False
    for action in range(ACTION_COUNT):
        afterstate, reward, changed = _apply_move(board, action)
        if not changed:
            continue
        future_value = _value(afterstate, weights, feature_cells)
        candidate = float(reward) + future_value
        if not has_action or candidate > best_value:
            best_value = candidate
            has_action = True
    return best_value


@njit(cache=True)
def _choose_expectimax_action(
    board: np.ndarray,
    weights: np.ndarray,
    feature_cells: np.ndarray,
    epsilon: float,
    legal_mask: np.ndarray,
) -> tuple[int, np.ndarray, int]:
    actions = np.empty(ACTION_COUNT, dtype=np.int8)
    afterstates = np.zeros((ACTION_COUNT, 16), dtype=np.uint8)
    rewards = np.zeros(ACTION_COUNT, dtype=np.int64)
    legal_count = 0

    for action in range(ACTION_COUNT):
        if not legal_mask[action]:
            continue
        afterstate, reward, changed = _apply_move(board, action)
        if changed:
            actions[legal_count] = action
            afterstates[legal_count] = afterstate
            rewards[legal_count] = reward
            legal_count += 1

    if legal_count == 0:
        return -1, np.zeros(16, dtype=np.uint8), 0
    if np.random.random() < epsilon:
        selected = np.random.randint(0, legal_count)
        return int(actions[selected]), afterstates[selected].copy(), int(rewards[selected])

    best_value = -1.0e30
    best_slot = 0
    tie_count = 0
    for slot in range(legal_count):
        afterstate = afterstates[slot]
        empty_count = 0
        for cell in range(16):
            if afterstate[cell] == 0:
                empty_count += 1

        if empty_count == 0:
            candidate = float(rewards[slot]) + _best_one_ply_value(
                afterstate, weights, feature_cells
            )
        else:
            expected_future = 0.0
            for cell in range(16):
                if afterstate[cell] != 0:
                    continue
                spawned = afterstate.copy()
                spawned[cell] = 1  # 90% chance of a 2 tile
                value_two = _best_one_ply_value(spawned, weights, feature_cells)
                spawned[cell] = 2  # 10% chance of a 4 tile
                value_four = _best_one_ply_value(spawned, weights, feature_cells)
                expected_future += 0.9 * value_two + 0.1 * value_four
            candidate = float(rewards[slot]) + expected_future / empty_count

        if candidate > best_value + 1.0e-5:
            best_value = candidate
            best_slot = slot
            tie_count = 1
        elif abs(candidate - best_value) <= 1.0e-5:
            tie_count += 1
            if np.random.randint(0, tie_count) == 0:
                best_slot = slot

    return int(actions[best_slot]), afterstates[best_slot].copy(), int(rewards[best_slot])


@njit(cache=True)
def _train_episode(
    weights: np.ndarray,
    feature_cells: np.ndarray,
    epsilon: float,
    learning_rate: float,
    max_steps: int,
) -> tuple[int, int, bool, int]:
    board = _new_board()
    previous_afterstate = np.zeros(16, dtype=np.uint8)
    has_previous = False
    total_score = 0
    won = False
    moves = 0
    legal_mask = np.ones(ACTION_COUNT, dtype=np.bool_)

    for _ in range(max_steps):
        action, afterstate, reward = _choose_action(
            board, weights, feature_cells, epsilon, legal_mask
        )
        if action < 0:
            if has_previous:
                _update_value(previous_afterstate, 0.0, weights, feature_cells, learning_rate)
            return total_score, 0, won, moves

        moves += 1
        total_score += reward
        max_exponent = 0
        for cell in range(16):
            if afterstate[cell] > max_exponent:
                max_exponent = afterstate[cell]
        if max_exponent >= WIN_EXPONENT:
            won = True

        next_board = afterstate.copy()
        _spawn_tile(next_board)
        terminal = not _has_moves(next_board)
        next_value = 0.0 if terminal else _value(afterstate, weights, feature_cells)

        if has_previous:
            _update_value(
                previous_afterstate,
                float(reward) + next_value,
                weights,
                feature_cells,
                learning_rate,
            )

        if terminal:
            _update_value(afterstate, 0.0, weights, feature_cells, learning_rate)
            return total_score, 1 << max_exponent, won, moves

        previous_afterstate = afterstate
        has_previous = True
        board = next_board

    if has_previous:
        _update_value(previous_afterstate, 0.0, weights, feature_cells, learning_rate)
    max_exponent = 0
    for cell in range(16):
        if board[cell] > max_exponent:
            max_exponent = board[cell]
    return total_score, 1 << max_exponent, won, moves


@njit(cache=True)
def _evaluate_game(
    weights: np.ndarray,
    feature_cells: np.ndarray,
    max_steps: int,
    search_depth: int,
) -> tuple[int, int, bool, int]:
    board = _new_board()
    total_score = 0
    won = False
    moves = 0
    legal_mask = np.ones(ACTION_COUNT, dtype=np.bool_)

    for _ in range(max_steps):
        if search_depth >= 2:
            action, afterstate, reward = _choose_expectimax_action(
                board, weights, feature_cells, 0.0, legal_mask
            )
        else:
            action, afterstate, reward = _choose_action(
                board, weights, feature_cells, 0.0, legal_mask
            )
        if action < 0:
            break
        moves += 1
        total_score += reward
        max_exponent = 0
        for cell in range(16):
            if afterstate[cell] > max_exponent:
                max_exponent = afterstate[cell]
        if max_exponent >= WIN_EXPONENT:
            won = True
        board = afterstate.copy()
        _spawn_tile(board)
        if not _has_moves(board):
            break

    max_exponent = 0
    for cell in range(16):
        if board[cell] > max_exponent:
            max_exponent = board[cell]
    return total_score, 1 << max_exponent, won, moves


class NtupleAgent:
    """Greedy afterstate policy backed by learned N-tuple value weights."""

    def __init__(self, weights_path: str | Path, search_depth: int = 2) -> None:
        if search_depth not in (1, 2):
            raise ValueError("search_depth must be 1 or 2")
        self.search_depth = search_depth
        self.weights_path = Path(weights_path)
        loaded = np.load(self.weights_path, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                self.weights = np.array(loaded["weights"], dtype=np.float32, copy=True)
        else:
            self.weights = loaded
        if self.weights.shape != TABLE_SHAPE or self.weights.dtype != np.float32:
            raise ValueError(
                f"expected float32 N-tuple weights shaped {TABLE_SHAPE}, "
                f"received {self.weights.dtype} {self.weights.shape}"
            )

    def select_action(
        self,
        observation: np.ndarray,
        legal_actions: Sequence[int],
        epsilon: float = 0.0,
    ) -> int:
        if not legal_actions:
            raise ValueError("cannot select an action when no legal moves exist")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        normalized = np.asarray(observation, dtype=np.float32).reshape(-1)
        if normalized.size != 16:
            raise ValueError("observation must contain exactly 16 board cells")
        board = np.rint(normalized * MAX_TUPLE_EXPONENT).astype(np.uint8)
        legal_mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        for action in legal_actions:
            if not 0 <= action < ACTION_COUNT:
                raise ValueError(f"invalid action id: {action}")
            legal_mask[action] = True
        if self.search_depth >= 2:
            action, _, _ = _choose_expectimax_action(
                board, self.weights, FEATURE_CELLS, epsilon, legal_mask
            )
        else:
            action, _, _ = _choose_action(
                board, self.weights, FEATURE_CELLS, epsilon, legal_mask
            )
        if action < 0:
            raise RuntimeError("the simulator and model disagree about legal moves")
        return action


def create_empty_weights() -> np.ndarray:
    """Allocate the four float32 tuple tables (about 256 MiB)."""
    return np.zeros(TABLE_SHAPE, dtype=np.float32)


def train_one_episode(
    weights: np.ndarray,
    epsilon: float,
    learning_rate: float = 0.1,
    max_steps: int = 20_000,
) -> tuple[int, int, bool, int]:
    return _train_episode(
        weights, FEATURE_CELLS, epsilon, learning_rate, max_steps
    )


def evaluate_games(
    weights: np.ndarray,
    game_count: int,
    seed: int,
    max_steps: int = 20_000,
    search_depth: int = 1,
) -> dict[str, int | float]:
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if search_depth not in (1, 2):
        raise ValueError("search_depth must be 1 or 2")
    scores = np.empty(game_count, dtype=np.int64)
    max_tiles = np.empty(game_count, dtype=np.int64)
    wins = 0
    moves = np.empty(game_count, dtype=np.int64)
    for game_index in range(game_count):
        seed_numba(seed + game_index)
        score, max_tile, won, move_count = _evaluate_game(
            weights, FEATURE_CELLS, max_steps, search_depth
        )
        scores[game_index] = score
        max_tiles[game_index] = max_tile
        moves[game_index] = move_count
        wins += int(won)
    best_game_index = max(
        range(game_count), key=lambda index: (max_tiles[index], scores[index])
    )
    return {
        "win_rate": wins / game_count,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "mean_max_tile": float(max_tiles.mean()),
        "mean_moves": float(moves.mean()),
        "best_tile": int(max_tiles[best_game_index]),
        "best_tile_score": int(scores[best_game_index]),
        "best_tile_seed": seed + best_game_index,
        "max_score": int(scores.max()),
    }
