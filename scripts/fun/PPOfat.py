
import os
import sys
import json
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# =============================================================================
# Local contest imports
# =============================================================================
sys.path.append(os.getcwd())

from engine.game import BomberEnv

# Baselines are optional during local development; keep graceful fallback.
BASELINE_IMPORT_ERRORS = []

try:
    from agent.tactical_rule_agent import TacticalRuleAgent
except Exception as e:
    TacticalRuleAgent = None
    BASELINE_IMPORT_ERRORS.append(("TacticalRuleAgent", repr(e)))

try:
    from agent.genius_rule_agent import GeniusRuleAgent
except Exception as e:
    GeniusRuleAgent = None
    BASELINE_IMPORT_ERRORS.append(("GeniusRuleAgent", repr(e)))

try:
    from agent.smarter_rule_agent import SmarterRuleAgent
except Exception as e:
    SmarterRuleAgent = None
    BASELINE_IMPORT_ERRORS.append(("SmarterRuleAgent", repr(e)))

try:
    from agent.box_farmer_agent import BoxFarmerAgent
except Exception as e:
    BoxFarmerAgent = None
    BASELINE_IMPORT_ERRORS.append(("BoxFarmerAgent", repr(e)))

try:
    from agent.simple_rule_agent import SimpleRuleAgent
except Exception as e:
    SimpleRuleAgent = None
    BASELINE_IMPORT_ERRORS.append(("SimpleRuleAgent", repr(e)))

try:
    from agent.random_agent import RandomAgent
except Exception as e:
    RandomAgent = None
    BASELINE_IMPORT_ERRORS.append(("RandomAgent", repr(e)))


# =============================================================================
# Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BOARD_SIZE = 13
INPUT_CHANNELS = 27
NUM_ACTIONS = 6
MAX_STEPS = 500
EXPLOSION_TIME_HORIZON = 8.0  # safe tiles stay at 1.0; lower values mean earlier explosions

INITIAL_GAMES = 800
DAGGER_ROUNDS = 2
DAGGER_GAMES_PER_ROUND = 200

TRAIN_SPLIT_MOD = 10  # seed % 10 == 0 -> validation
CHUNK_SIZE = 2048
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRAD_CLIP_NORM = 1.0

TRAIN_DIR = "bc_train_chunks"
VAL_DIR = "bc_val_chunks"
MODEL_PATH = "model_bc.pth"
BEST_MODEL_PATH = "model_bc_best.pth"
MANIFEST_NAME = "manifest.json"

# Teacher settings
USE_ENSEMBLE_TEACHER = True
TEACHER_VOTE_MODE = "weighted"  # weighted | majority | tactical_priority
TEACHER_RANDOM_TEMPERATURE = 0.10  # small noise in tie-breaks

# Augmentation
AUGMENT_FLIP_PROB = 1.0

# =============================================================================
# Seeding
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =============================================================================
# Board utilities
# =============================================================================
# Game actions:
#   0 STOP, 1 LEFT, 2 RIGHT, 3 UP, 4 DOWN, 5 PLACE_BOMB
MOVES = {
    0: (0, 0),
    1: (0, -1),   # LEFT
    2: (0, 1),    # RIGHT
    3: (-1, 0),   # UP
    4: (1, 0),    # DOWN
}


def next_pos(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    dr, dc = MOVES[int(action)]
    return pos[0] + dr, pos[1] + dc


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)


def bomb_positions_set(bombs: np.ndarray) -> set:
    if bombs is None or len(bombs) == 0:
        return set()
    return {(int(b[0]), int(b[1])) for b in bombs}


def bomb_radius_for_owner(players: np.ndarray, owner: int) -> int:
    if 0 <= owner < len(players) and int(players[owner][2]) == 1:
        return 1 + int(players[owner][4])
    return 1


def blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set:
    tiles = {(bx, by)}
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for d in range(1, radius + 1):
            r, c = bx + dr * d, by + dc * d
            if not in_bounds(r, c):
                break
            cell = int(grid[r, c])
            if cell == 1:
                break
            tiles.add((r, c))
            if cell == 2:
                break
    return tiles



def bomb_effective_explosion_times(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    """
    Resolve chain reactions and return the effective explosion timer for each bomb.
    If bomb A explodes earlier and its blast reaches bomb B, then B inherits A's
    earlier explosion time.
    """
    if bombs is None or len(bombs) == 0:
        return np.zeros((0,), dtype=np.int32)

    n = len(bombs)
    times = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    blasts: List[set] = []
    for i in range(n):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        blasts.append(blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius))

    q = deque(range(n))
    in_queue = [True] * n
    while q:
        i = q.popleft()
        in_queue[i] = False
        ti = int(times[i])
        if ti < 0:
            ti = 0
        for j in range(n):
            if i == j:
                continue
            bj = (int(bombs[j][0]), int(bombs[j][1]))
            if bj in blasts[i] and int(times[j]) > ti:
                times[j] = ti
                if not in_queue[j]:
                    q.append(j)
                    in_queue[j] = True
    return times


def explosion_time_plane(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    horizon: float = EXPLOSION_TIME_HORIZON,
) -> np.ndarray:
    """
    Per-tile earliest explosion time plane.

    Encoding:
      - safe tiles: 1.0
      - threatened tiles: min(explosion_time, horizon) / horizon
        so smaller means sooner and safe remains the largest value.
    """
    plane = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane

    times = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t = float(max(0, int(times[i])))
        norm_t = min(t, horizon) / horizon if horizon > 0 else 0.0
        for r, c in blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius):
            if norm_t < plane[r, c]:
                plane[r, c] = norm_t
    return plane


def danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, timer_threshold: int = 1) -> np.ndarray:
    """
    Binary danger plane derived from the chain-reaction-aware explosion times.
    """
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger

    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / float(EXPLOSION_TIME_HORIZON) if EXPLOSION_TIME_HORIZON > 0 else 0.0
    danger[plane <= threshold] = 1.0
    return danger


def immediate_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    """Tiles that are exploding now or on the next step."""
    return danger_plane(grid, players, bombs, timer_threshold=1)


def chain_danger_plane(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    chain_horizon: int = 3,
) -> np.ndarray:
    """
    Tiles that are dangerous because a bomb is accelerated by chain reaction,
    but not already counted as immediate danger.
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane

    original = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    effective = bomb_effective_explosion_times(grid, players, bombs)

    for i in range(len(bombs)):
        # Must be chain-accelerated and not immediate.
        if int(effective[i]) <= 1:
            continue
        if int(effective[i]) > chain_horizon:
            continue
        if int(effective[i]) >= int(original[i]):
            continue

        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        for r, c in blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius):
            plane[r, c] = 1.0
    return plane


def future_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, horizon: float = EXPLOSION_TIME_HORIZON) -> np.ndarray:
    """Soft danger map: higher means sooner explosion risk."""
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane

    effective = bomb_effective_explosion_times(grid, players, bombs)
    denom = float(max(1.0, horizon))
    for i in range(len(bombs)):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t = float(max(0, int(effective[i])))
        score = 1.0 - min(t, denom) / denom
        if score <= 0:
            continue
        for r, c in blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius):
            plane[r, c] = max(plane[r, c], score)
    return plane



def tile_earliest_explosion_times(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    """
    Exact earliest explosion time per tile after chain reactions.
    Safe tiles are marked with a large sentinel value.
    """
    times = np.full((BOARD_SIZE, BOARD_SIZE), 9999, dtype=np.int32)
    if bombs is None or len(bombs) == 0:
        return times

    eff = bomb_effective_explosion_times(grid, players, bombs)
    for i, b in enumerate(bombs):
        owner = int(b[3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t = int(max(0, eff[i]))
        for r, c in blast_tiles(grid, int(b[0]), int(b[1]), radius):
            if t < times[r, c]:
                times[r, c] = t
    return times


def bomb_pressure_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    """
    Enemy bomb pressure from bombs that enemies can plausibly place from their
    current positions right now.
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None:
        bombs = np.zeros((0, 4), dtype=np.int8)

    for pid in range(4):
        if pid == my_id or pid >= len(players):
            continue
        if int(players[pid][2]) != 1:
            continue
        bombs_left = int(players[pid][3])
        if bombs_left <= 0:
            continue

        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c):
            continue

        # Current-position bomb threat.
        radius = 1 + int(players[pid][4])
        for x, y in blast_tiles(grid, r, c, radius):
            plane[x, y] = max(plane[x, y], 1.0)

    return plane


def future_bomb_pressure_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    """
    Approximate pressure if enemies move one step and then are able to bomb on a later step.
    This is intentionally soft and low-cost.
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None:
        bombs = np.zeros((0, 4), dtype=np.int8)

    blocked = bomb_positions_set(bombs)
    for pid in range(4):
        if pid == my_id or pid >= len(players):
            continue
        if int(players[pid][2]) != 1:
            continue
        bombs_left = int(players[pid][3])
        if bombs_left <= 0:
            continue

        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c):
            continue

        radius = 1 + int(players[pid][4])
        # Likely next-step positions if the enemy chooses to reposition.
        candidate_tiles = [(r, c)]
        for a in (1, 2, 3, 4):
            nr, nc = next_pos((r, c), a)
            if passable(grid, nr, nc) and (nr, nc) not in blocked:
                candidate_tiles.append((nr, nc))

        for pr, pc in candidate_tiles:
            for x, y in blast_tiles(grid, pr, pc, radius):
                # softer weight than an immediate bomb threat
                plane[x, y] = max(plane[x, y], 0.5)

    return plane


def bottleneck_risk_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    """
    High when my current position has very few safe exits and those exits are fragile.
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    blocked = bomb_positions_set(bombs)
    explosion_times = tile_earliest_explosion_times(grid, players, bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)

    # A tile is risky if it has <= 1 safe neighboring escape and is close to an explosion.
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not passable(grid, r, c):
                continue
            if (r, c) in blocked:
                continue

            exits = 0
            fragile = 0
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if not passable(grid, nr, nc):
                    continue
                if (nr, nc) in blocked:
                    continue
                exits += 1
                if danger_now[nr, nc] > 0.0 or explosion_times[nr, nc] <= 2:
                    fragile += 1

            if exits == 0:
                score = 1.0
            elif exits == 1:
                score = 0.85 if fragile > 0 else 0.65
            elif exits == 2:
                score = 0.4 if fragile >= 2 else 0.2
            else:
                score = 0.0

            # Stronger around my current position and nearby cells.
            manhattan = abs(r - my_r) + abs(c - my_c)
            if manhattan <= 1:
                score = max(score, 0.75)
            elif manhattan <= 2:
                score = max(score, 0.35)

            plane[r, c] = score

    return plane


def escape_margin_from_position(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    start: Tuple[int, int],
    max_depth: int = 6,
) -> float:
    """
    Largest positive margin between when a tile explodes and the time needed to reach it.
    > 0 means there exists at least one tile reachable before it explodes.
    """
    explosion_times = tile_earliest_explosion_times(grid, players, bombs)
    blocked = bomb_positions_set(bombs)

    q = deque([(start, 0)])
    seen = {start}
    best_margin = -9999

    while q:
        pos, dist = q.popleft()
        t_exp = int(explosion_times[pos[0], pos[1]])
        margin = t_exp - dist
        if margin > best_margin:
            best_margin = margin
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if npos in seen:
                continue
            if npos in blocked:
                continue
            if not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))

    if best_margin < -1000:
        return -1.0
    return float(best_margin)


def time_safe_escape_score(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> float:
    """
    Normalized escape score from the agent's current position.
    """
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return 0.0
    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    margin = escape_margin_from_position(grid, players, bombs, my_pos, max_depth=6)
    if margin <= 0:
        return 0.0
    return float(np.clip(margin / 6.0, 0.0, 1.0))


def should_place_bomb_here(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, pos: Tuple[int, int]) -> bool:
    """
    Cheap survival check for a hypothetical bomb at pos.
    We only require at least one escape route before the bomb timer window closes.
    """
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return False
    if not passable(grid, pos[0], pos[1]):
        return False

    my_radius = 1 + int(players[my_id][4])
    blocked = bomb_positions_set(bombs)

    # Candidate blast area if I bomb this tile.
    blast = blast_tiles(grid, pos[0], pos[1], my_radius)

    # Need an exit that is not in blast, not blocked, and reachable in time.
    for a in (1, 2, 3, 4):
        nr, nc = next_pos(pos, a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked:
            continue
        if (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, bombs, (nr, nc), max_depth=6) > 0:
            return True
    return False



def safe_to_bomb_plane(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    my_id: int,
) -> np.ndarray:
    """
    Mark my current tile if placing a bomb there looks both useful and survivable.
    This is intentionally compact so the model sees a clear bomb-safety signal.
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    if not in_bounds(my_r, my_c):
        return plane

    # Must be a place where a bomb can actually be dropped.
    blocked = bomb_positions_set(bombs)
    if (my_r, my_c) in blocked:
        return plane

    bomb_radius = 1 + int(players[my_id][4])
    blast = blast_tiles(grid, my_r, my_c, bomb_radius)

    # Good if it is tactically meaningful.
    enemy_positions = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(4)
        if i != my_id and i < len(players) and int(players[i][2]) == 1
    }
    hit_boxes = any(int(grid[x, y]) == 2 for x, y in blast)
    hit_enemy = any((x, y) in enemy_positions for x, y in blast)
    if not (hit_boxes or hit_enemy):
        return plane

    # Must have at least one escape route that remains valid before explosions happen.
    safe_exit = False
    for a in (1, 2, 3, 4):
        nr, nc = next_pos((my_r, my_c), a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked:
            continue
        if (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, bombs, (nr, nc), max_depth=6) > 0:
            safe_exit = True
            break

    if safe_exit:
        plane[my_r, my_c] = 1.0
    return plane


def immediate_blast_tiles_if_placed(grid: np.ndarray, pos: Tuple[int, int], radius: int) -> set:
    return blast_tiles(grid, pos[0], pos[1], radius)


def bfs_distance_to_targets(
    grid: np.ndarray,
    start: Tuple[int, int],
    targets: set,
    bombs: np.ndarray,
    max_depth: int = 64,
) -> Optional[int]:
    if not targets:
        return None

    blocked = bomb_positions_set(bombs)
    q = deque([(start, 0)])
    seen = {start}

    while q:
        pos, dist = q.popleft()
        if pos in targets:
            return dist
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if npos in seen:
                continue
            if npos in blocked:
                continue
            if not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return None


def bfs_reachable_count(grid: np.ndarray, start: Tuple[int, int], bombs: np.ndarray, max_depth: int = 3) -> int:
    blocked = bomb_positions_set(bombs)
    q = deque([(start, 0)])
    seen = {start}
    count = 0

    while q:
        pos, dist = q.popleft()
        if dist > 0:
            count += 1
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if npos in seen:
                continue
            if npos in blocked:
                continue
            if not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return count


def bfs_escape_available(
    grid: np.ndarray,
    start: Tuple[int, int],
    players: np.ndarray,
    bombs: np.ndarray,
    max_depth: int = 6,
) -> int:
    blocked = bomb_positions_set(bombs)
    danger = danger_plane(grid, players, bombs, timer_threshold=1)
    q = deque([(start, 0)])
    seen = {start}

    while q:
        pos, dist = q.popleft()
        if dist > 0 and danger[pos[0], pos[1]] == 0.0:
            return 1
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if npos in seen:
                continue
            if npos in blocked:
                continue
            if not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return 0


def norm_dist(d: Optional[int], cap: float = 24.0) -> float:
    if d is None:
        return 1.0
    return float(min(d, cap)) / cap


def normalize_scalar(x: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(np.clip(x / denom, 0.0, 1.0))


def legal_actions(grid: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int], bombs_left: int) -> List[int]:
    moves = [0]
    blocked = bomb_positions_set(bombs)
    for a in (1, 2, 3, 4):
        nr, nc = next_pos(my_pos, a)
        if passable(grid, nr, nc) and (nr, nc) not in blocked:
            moves.append(a)
    if bombs_left > 0 and my_pos not in blocked:
        moves.append(5)
    return moves


def movement_actions_from_legal(legal: Iterable[int]) -> List[int]:
    """Return only movement actions, never bomb placement."""
    return [int(a) for a in legal if int(a) in (1, 2, 3, 4)]


# =============================================================================
# Observation encoding
# =============================================================================

def encode_obs(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int) -> torch.Tensor:
    """
    Updated 27-channel encoder.

    0  wall
    1  box
    2  grass
    3  radius item
    4  capacity item
    5  player 0 position
    6  player 1 position
    7  player 2 position
    8  player 3 position
    9  earliest explosion time plane
    10 immediate danger plane
    11 chain-reaction danger plane
    12 future danger plane
    13 my position
    14 bombs_left normalized plane
    15 bomb timer heatmap
    16 bomb radius heatmap
    17 BFS distance to nearest item
    18 BFS distance to nearest enemy
    19 local reachability plane
    20 time-aware escape score plane
    21 safe-to-bomb plane
    22 bomb count normalized plane
    23 step ratio plane
    24 enemy current bomb pressure
    25 enemy future bomb pressure
    26 bottleneck risk
    """
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Static map
    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    # Player positions by id
    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                state[5 + pid, r, c] = 1.0

    # Bomb danger system
    state[9] = explosion_time_plane(grid, players, bombs)
    state[10] = immediate_danger_plane(grid, players, bombs)
    state[11] = chain_danger_plane(grid, players, bombs)
    state[12] = future_danger_plane(grid, players, bombs)

    # Ego features
    me_alive = 0
    my_pos = (0, 0)
    bombs_left = 0
    if my_id < len(players) and int(players[my_id][2]) == 1:
        me_alive = 1
        mr, mc = int(players[my_id][0]), int(players[my_id][1])
        my_pos = (mr, mc)
        if in_bounds(mr, mc):
            state[13, mr, mc] = 1.0
        bombs_left = int(players[my_id][3])

    state[14].fill(normalize_scalar(bombs_left, 5.0))

    if bombs is not None and len(bombs) > 0:
        eff_times = bomb_effective_explosion_times(grid, players, bombs)
        for i, b in enumerate(bombs):
            r, c = int(b[0]), int(b[1])
            if not in_bounds(r, c):
                continue
            t = max(int(eff_times[i]), 1)
            state[15, r, c] = max(state[15, r, c], 1.0 / float(t))
            owner = int(b[3]) if len(b) > 3 else -1
            state[16, r, c] = max(state[16, r, c], normalize_scalar(bomb_radius_for_owner(players, owner), 6.0))
    else:
        state[15].fill(0.0)
        state[16].fill(0.0)

    if me_alive:
        item_pos = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
        enemy_pos = {
            (int(players[i][0]), int(players[i][1]))
            for i in range(4)
            if i != my_id and i < len(players) and int(players[i][2]) == 1
        }

        d_item = bfs_distance_to_targets(grid, my_pos, item_pos, bombs)
        d_enemy = bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs)

        state[17].fill(norm_dist(d_item))
        state[18].fill(norm_dist(d_enemy))
        state[19].fill(normalize_scalar(bfs_reachable_count(grid, my_pos, bombs, max_depth=3), 20.0))
        state[20].fill(time_safe_escape_score(grid, players, bombs, my_id))
        state[21] = safe_to_bomb_plane(grid, players, bombs, my_id)
    else:
        state[17].fill(1.0)
        state[18].fill(1.0)
        state[19].fill(0.0)
        state[20].fill(0.0)
        state[21].fill(0.0)

    state[22].fill(normalize_scalar(len(bombs) if bombs is not None else 0, 10.0))
    state[23].fill(normalize_scalar(step, float(MAX_STEPS)))
    state[24] = bomb_pressure_plane(grid, players, bombs, my_id)
    state[25] = future_bomb_pressure_plane(grid, players, bombs, my_id)
    state[26] = bottleneck_risk_plane(grid, players, bombs, my_id)

    return torch.from_numpy(state)


# =============================================================================
# Teacher ensemble

# =============================================================================

class _FallbackRuleAgent:
    """
    Tiny local fallback if baseline imports are unavailable.
    Not intended to be strong, only to keep the trainer runnable.
    """
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: Dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0
        r, c = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left = int(players[self.agent_id][3])

        danger = danger_plane(grid, players, bombs, timer_threshold=1)
        if danger[r, c] > 0:
            moves = []
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if passable(grid, nr, nc) and danger[nr, nc] == 0 and (nr, nc) not in bomb_positions_set(bombs):
                    moves.append(a)
            if moves:
                return int(random.choice(moves))
            return 0

        items = {(int(x), int(y)) for x, y in np.argwhere((grid == 3) | (grid == 4))}
        if items:
            best = 0
            best_d = 10**9
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if passable(grid, nr, nc) and (nr, nc) not in bomb_positions_set(bombs):
                    d = min(abs(nr - ir) + abs(nc - ic) for ir, ic in items)
                    if d < best_d:
                        best_d = d
                        best = a
            if best != 0:
                return int(best)

        if bombs_left > 0:
            return 5
        return 0


def _maybe_make(cls, agent_id: int):
    if cls is None:
        return _FallbackRuleAgent(agent_id)
    return cls(agent_id)


@dataclass
class TeacherWeights:
    tactical: float = 3.0
    genius: float = 2.5
    smarter: float = 2.0
    box_farmer: float = 1.0
    simple: float = 0.75
    random: float = 0.25



class TeacherEnsemble:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.tactical = _maybe_make(TacticalRuleAgent, agent_id)
        self.genius = _maybe_make(GeniusRuleAgent, agent_id)
        self.smarter = _maybe_make(SmarterRuleAgent, agent_id)
        self.box_farmer = _maybe_make(BoxFarmerAgent, agent_id)
        self.simple = _maybe_make(SimpleRuleAgent, agent_id)
        self.random = _maybe_make(RandomAgent, agent_id)
        self.weights = TeacherWeights()

    def _collect_actions(self, obs: Dict) -> Dict[str, int]:
        return {
            "tactical": int(self.tactical.act(obs)),
            "genius": int(self.genius.act(obs)),
            "smarter": int(self.smarter.act(obs)),
            "box_farmer": int(self.box_farmer.act(obs)),
            "simple": int(self.simple.act(obs)),
            "random": int(self.random.act(obs)),
        }

    def _weighted_vote(self, actions: Dict[str, int], legal: Optional[set] = None) -> int:
        score = Counter()
        score[actions["tactical"]] += self.weights.tactical
        score[actions["genius"]] += self.weights.genius
        score[actions["smarter"]] += self.weights.smarter
        score[actions["box_farmer"]] += self.weights.box_farmer
        score[actions["simple"]] += self.weights.simple
        score[actions["random"]] += self.weights.random

        if legal is not None:
            for a in list(score.keys()):
                if a not in legal:
                    score[a] -= 10.0

        priority = [
            actions["tactical"],
            actions["genius"],
            actions["smarter"],
            actions["box_farmer"],
            actions["simple"],
            actions["random"],
        ]
        best_score = max(score.values())
        candidates = [a for a, s in score.items() if abs(s - best_score) < 1e-9]
        for preferred in priority:
            if preferred in candidates:
                return int(preferred)
        return int(candidates[0])

    def _move_score(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int]) -> float:
        if not passable(grid, pos[0], pos[1]):
            return -1e9
        blocked = bomb_positions_set(bombs)
        if pos in blocked:
            return -1e9

        # Hard safety first.
        margin = escape_margin_from_position(grid, players, bombs, pos, max_depth=6)
        if margin <= 0:
            return -1000.0

        score = 2.0 * margin
        if danger_plane(grid, players, bombs, timer_threshold=1)[pos[0], pos[1]] > 0.0:
            score -= 5.0
        if bomb_pressure_plane(grid, players, bombs, self.agent_id)[pos[0], pos[1]] > 0.0:
            score -= 2.0
        if future_bomb_pressure_plane(grid, players, bombs, self.agent_id)[pos[0], pos[1]] > 0.0:
            score -= 1.0

        # Prefer routes that still keep options open.
        reachable = bfs_reachable_count(grid, pos, bombs, max_depth=3)
        score += 0.05 * float(reachable)
        return float(score)

    def _best_escape_action(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, legal: set, my_pos: Tuple[int, int]) -> int:
        best_action = 0
        best_score = -1e18
        for a in movement_actions_from_legal(legal):
            nr, nc = next_pos(my_pos, a)
            s = self._move_score(grid, players, bombs, (nr, nc))
            if s > best_score:
                best_score = s
                best_action = int(a)
        return int(best_action)

    def act(self, obs: Dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        r, c = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left = int(players[self.agent_id][3])
        legal = set(legal_actions(grid, bombs, (r, c), bombs_left))

        actions = self._collect_actions(obs)

        danger = danger_plane(grid, players, bombs, timer_threshold=1)
        current_pressure = bomb_pressure_plane(grid, players, bombs, self.agent_id)
        future_pressure = future_bomb_pressure_plane(grid, players, bombs, self.agent_id)
        bottleneck = bottleneck_risk_plane(grid, players, bombs, self.agent_id)

        # If we are in danger or in a brittle corridor, prefer the safest move.
        if danger[r, c] > 0.0 or bottleneck[r, c] > 0.65 or current_pressure[r, c] > 0.0:
            safe_move = self._best_escape_action(grid, players, bombs, legal, (r, c))
            if safe_move in legal and safe_move != 0:
                return int(safe_move)

        # Only consider bomb placement if it is tactically useful and survivable.
        if 5 in legal and should_place_bomb_here(grid, players, bombs, self.agent_id, (r, c)):
            # When the ensemble strongly prefers bombing, let it through.
            if actions["tactical"] == 5 or actions["genius"] == 5 or actions["smarter"] == 5:
                return 5
            # If we are near boxes/enemies and not under threat, bomb is often the right move.
            blast = blast_tiles(grid, r, c, 1 + int(players[self.agent_id][4]))
            hit_boxes = any(int(grid[x, y]) == 2 for x, y in blast)
            hit_enemy = any(
                (x, y) in {
                    (int(players[i][0]), int(players[i][1]))
                    for i in range(4)
                    if i != self.agent_id and i < len(players) and int(players[i][2]) == 1
                }
                for x, y in blast
            )
            if (hit_boxes or hit_enemy) and danger[r, c] == 0.0 and current_pressure[r, c] == 0.0:
                return 5

        # Heuristic weighting tweaks.
        box_count = int(np.sum(grid == 2))
        if box_count >= 18:
            self.weights.box_farmer = 2.2
        else:
            self.weights.box_farmer = 1.2

        alive_cnt = int(np.sum(players[:, 2])) if len(players) else 0
        if alive_cnt <= 2:
            self.weights.tactical = 3.5
            self.weights.genius = 2.8
            self.weights.smarter = 2.0
        else:
            self.weights.tactical = 3.0
            self.weights.genius = 2.5
            self.weights.smarter = 2.0

        # If enemy pressure is high, make the ensemble more conservative.
        if current_pressure[r, c] > 0.0 or future_pressure[r, c] > 0.0:
            self.weights.random = 0.05
            self.weights.simple = 0.5
        else:
            self.weights.random = 0.25
            self.weights.simple = 0.75

        # Prefer the safest legal move if the vote picks an unsafe action while threat is active.
        vote = self._weighted_vote(actions, legal=legal)
        if vote == 5 and 5 in legal and not should_place_bomb_here(grid, players, bombs, self.agent_id, (r, c)):
            vote = self._best_escape_action(grid, players, bombs, legal, (r, c))
            if vote == 0 and 0 in legal:
                return 0

        if vote in (1, 2, 3, 4) and danger[r, c] > 0.0:
            nr, nc = next_pos((r, c), vote)
            if not passable(grid, nr, nc) or danger[nr, nc] > 0.0:
                alt = self._best_escape_action(grid, players, bombs, legal, (r, c))
                if alt in legal:
                    return int(alt)

        return int(vote)


# =============================================================================
# Model

# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class BomberNet(nn.Module):
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_actions: int = NUM_ACTIONS, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.1),
            ResidualBlock(width, dropout=0.1),
            ResidualBlock(width, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


# =============================================================================
# Augmentation
# =============================================================================
def remap_action_horizontal(action: int) -> int:
    # Mirror across vertical axis: LEFT <-> RIGHT
    return {1: 2, 2: 1, 3: 3, 4: 4, 0: 0, 5: 5}.get(int(action), int(action))


def remap_action_vertical(action: int) -> int:
    # Mirror across horizontal axis: UP <-> DOWN
    return {3: 4, 4: 3, 1: 1, 2: 2, 0: 0, 5: 5}.get(int(action), int(action))


def augment_tensor_and_action(state: torch.Tensor, action: int) -> Tuple[torch.Tensor, int]:
    if random.random() > AUGMENT_FLIP_PROB:
        return state, int(action)

    p = random.random()
    if p < 0.33:
        state = torch.flip(state, dims=[2])  # horizontal flip
        action = remap_action_horizontal(action)
    elif p < 0.66:
        state = torch.flip(state, dims=[1])  # vertical flip
        action = remap_action_vertical(action)
    else:
        state = torch.flip(state, dims=[1, 2])  # 180 degree
        action = remap_action_vertical(remap_action_horizontal(action))
    return state, int(action)


# =============================================================================
# Chunk helpers
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def manifest_path(chunk_dir: str) -> str:
    return os.path.join(chunk_dir, MANIFEST_NAME)


def load_manifest(chunk_dir: str) -> Dict:
    path = manifest_path(chunk_dir)
    if not os.path.exists(path):
        return {"version": 1, "chunks": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(chunk_dir: str, manifest: Dict) -> None:
    with open(manifest_path(chunk_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def flush_chunk(chunk_dir: str, chunk_idx: int, states: List[np.ndarray], actions: List[int], seeds: List[int]) -> Dict:
    if not states:
        return {}
    states_np = np.stack(states, axis=0).astype(np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    seeds_np = np.array(seeds, dtype=np.int64)
    hist = np.bincount(actions_np, minlength=NUM_ACTIONS).astype(int).tolist()

    filename = f"chunk_{chunk_idx:05d}.npz"
    file_path = os.path.join(chunk_dir, filename)
    np.savez_compressed(file_path, states=states_np, actions=actions_np, seeds=seeds_np)

    return {
        "file": filename,
        "count": int(len(actions_np)),
        "action_hist": hist,
        "seed_min": int(seeds_np.min()) if len(seeds_np) else None,
        "seed_max": int(seeds_np.max()) if len(seeds_np) else None,
    }


# =============================================================================
# Dataset
# =============================================================================
class ChunkedBomberIterableDataset(IterableDataset):
    def __init__(self, chunk_dir: str, augment: bool, shuffle_chunks: bool, shuffle_within_chunk: bool, seed: int):
        super().__init__()
        self.chunk_dir = chunk_dir
        self.augment = augment
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = seed
        self.manifest = load_manifest(chunk_dir)
        self.chunks = list(self.manifest.get("chunks", []))
        self.counts = [int(c.get("count", 0)) for c in self.chunks]
        self.total_len = int(sum(self.counts))

    def __len__(self) -> int:
        return self.total_len

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers

        rng = np.random.default_rng(self.seed + worker_id * 1337)
        chunk_indices = np.arange(len(self.chunks))
        if self.shuffle_chunks:
            rng.shuffle(chunk_indices)

        # worker sharding
        chunk_indices = chunk_indices[worker_id::num_workers]

        for chunk_idx in chunk_indices:
            chunk_meta = self.chunks[int(chunk_idx)]
            file_path = os.path.join(self.chunk_dir, chunk_meta["file"])
            data = np.load(file_path)
            states = data["states"]
            actions = data["actions"]

            idxs = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(idxs)

            for i in idxs:
                state = torch.from_numpy(states[int(i)]).float()
                action = int(actions[int(i)])
                if self.augment:
                    state, action = augment_tensor_and_action(state, action)
                yield state, torch.tensor(action, dtype=torch.long)


# =============================================================================
# Class weights
# =============================================================================
def compute_class_weights(chunk_dir: str) -> torch.Tensor:
    manifest = load_manifest(chunk_dir)
    total = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for chunk in manifest.get("chunks", []):
        total += np.array(chunk.get("action_hist", [0] * NUM_ACTIONS), dtype=np.float64)

    total = np.maximum(total, 1.0)
    weights = total.sum() / total
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Opponent setup
# =============================================================================
def build_opponents(controlled_id: int, game_seed: int) -> Dict[int, object]:
    rng = random.Random(game_seed)
    pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent] if cls is not None]
    if not pool:
        pool = [_FallbackRuleAgent]

    chosen = []
    for _ in range(3):
        chosen.append(rng.choice(pool))
    other_ids = [pid for pid in range(4) if pid != controlled_id]
    opponents = {}
    for pid, cls in zip(other_ids, chosen):
        opponents[pid] = cls(pid)
    return opponents


# =============================================================================
# Data collection
# =============================================================================
def collect_one_step(obs: Dict, teacher: TeacherEnsemble, my_id: int, step: int) -> Tuple[np.ndarray, int]:
    state = encode_obs(obs["map"], obs["players"], obs["bombs"], my_id, step).numpy().astype(np.float32)
    action = int(teacher.act(obs))
    return state, action


def collect_initial_data(train_dir: str, val_dir: str, num_games: int) -> None:
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    train_manifest = load_manifest(train_dir)
    val_manifest = load_manifest(val_dir)
    train_chunk_idx = len(train_manifest.get("chunks", []))
    val_chunk_idx = len(val_manifest.get("chunks", []))

    train_buf_states, train_buf_actions, train_buf_seeds = [], [], []
    val_buf_states, val_buf_actions, val_buf_seeds = [], [], []

    for game_idx in range(num_games):
        seed = SEED + game_idx
        controlled_id = game_idx % 4
        split = "val" if (seed % TRAIN_SPLIT_MOD == 0) else "train"

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherEnsemble(controlled_id)
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            state_np, expert_action = collect_one_step(obs, teacher, controlled_id, step)

            if split == "train":
                train_buf_states.append(state_np)
                train_buf_actions.append(expert_action)
                train_buf_seeds.append(seed)
            else:
                val_buf_states.append(state_np)
                val_buf_actions.append(expert_action)
                val_buf_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = expert_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            if split == "train" and len(train_buf_states) >= CHUNK_SIZE:
                entry = flush_chunk(train_dir, train_chunk_idx, train_buf_states, train_buf_actions, train_buf_seeds)
                if entry:
                    train_manifest["chunks"].append(entry)
                    save_manifest(train_dir, train_manifest)
                    train_chunk_idx += 1
                train_buf_states.clear()
                train_buf_actions.clear()
                train_buf_seeds.clear()

            if split == "val" and len(val_buf_states) >= CHUNK_SIZE:
                entry = flush_chunk(val_dir, val_chunk_idx, val_buf_states, val_buf_actions, val_buf_seeds)
                if entry:
                    val_manifest["chunks"].append(entry)
                    save_manifest(val_dir, val_manifest)
                    val_chunk_idx += 1
                val_buf_states.clear()
                val_buf_actions.clear()
                val_buf_seeds.clear()

        if (game_idx + 1) % 100 == 0:
            train_count = sum(c["count"] for c in train_manifest.get("chunks", [])) + len(train_buf_actions)
            val_count = sum(c["count"] for c in val_manifest.get("chunks", [])) + len(val_buf_actions)
            print(f"Collected {game_idx + 1}/{num_games} games | train={train_count} | val={val_count}", flush=True)

    if train_buf_states:
        entry = flush_chunk(train_dir, train_chunk_idx, train_buf_states, train_buf_actions, train_buf_seeds)
        if entry:
            train_manifest["chunks"].append(entry)
    if val_buf_states:
        entry = flush_chunk(val_dir, val_chunk_idx, val_buf_states, val_buf_actions, val_buf_seeds)
        if entry:
            val_manifest["chunks"].append(entry)

    save_manifest(train_dir, train_manifest)
    save_manifest(val_dir, val_manifest)


def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int) -> int:
    ensure_dir(out_dir)
    model.eval()

    out_manifest = load_manifest(out_dir)
    chunk_idx = len(out_manifest.get("chunks", []))
    buf_states, buf_actions, buf_seeds = [], [], []
    collected = 0

    def flush_buffer():
        nonlocal chunk_idx, collected
        if not buf_states:
            return
        entry = flush_chunk(out_dir, chunk_idx, buf_states, buf_actions, buf_seeds)
        if entry:
            out_manifest["chunks"].append(entry)
            save_manifest(out_dir, out_manifest)
            collected += entry["count"]
            chunk_idx += 1
        buf_states.clear()
        buf_actions.clear()
        buf_seeds.clear()

    for game_idx in range(num_games):
        seed = 100000 + SEED + game_idx
        controlled_id = game_idx % 4

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherEnsemble(controlled_id)
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(state)
                student_action = int(torch.argmax(logits, dim=1).item())

            expert_action = int(teacher.act(obs))

            # collect disagreements and risky deviations
            if student_action != expert_action or (student_action == 0 and expert_action != 0):
                buf_states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
                buf_actions.append(expert_action)
                buf_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = student_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            if len(buf_states) >= CHUNK_SIZE:
                flush_buffer()

        if (game_idx + 1) % 50 == 0:
            print(f"DAgger {game_idx + 1}/{num_games} games | new samples={collected + len(buf_actions)}", flush=True)

    flush_buffer()
    return collected


# =============================================================================
# Training
# =============================================================================
def build_loaders(train_dir: str, val_dir: str):
    train_ds = ChunkedBomberIterableDataset(
        train_dir,
        augment=True,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        seed=SEED,
    )
    val_ds = ChunkedBomberIterableDataset(
        val_dir,
        augment=False,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        seed=SEED,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation samples found in {val_dir}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    class_weights = compute_class_weights(train_dir).to(DEVICE)
    return train_loader, val_loader, class_weights


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (states, actions) in enumerate(loader):
        states = states.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(states)
        loss = criterion(logits, actions)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item()) * states.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == actions).sum().item())
        total_count += int(states.size(0))

        if batch_idx % 50 == 0:
            print(f"  batch={batch_idx}", flush=True)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def train_policy_model(train_dir: str, val_dir: str, init_model_path: Optional[str] = None, lr: float = LEARNING_RATE):
    train_loader, val_loader, class_weights = build_loaders(train_dir, val_dir)

    model = BomberNet(INPUT_CHANNELS).to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        state = torch.load(init_model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_left = PATIENCE
            print(f"  -> saved best model to {BEST_MODEL_PATH}", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  -> early stopping", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}", flush=True)
    return model


# =============================================================================
# Optional quick eval
# =============================================================================
def quick_eval_against_baselines(model: nn.Module, num_games: int = 20) -> None:
    model.eval()
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(num_games):
        seed = 200000 + SEED + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            with torch.no_grad():
                state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
                my_pos = (int(obs["players"][controlled_id][0]), int(obs["players"][controlled_id][1]))
                bombs_left = int(obs["players"][controlled_id][3])
                legal_mask = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bombs_left)
                shielded_mask = _shielded_legal_mask(
                    obs["map"],
                    obs["players"],
                    obs["bombs"],
                    controlled_id,
                    legal_mask,
                )
                logits = model(state)
                masked_logits = logits.clone()
                mask = torch.tensor(shielded_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
                masked_logits[~mask] = -1e9
                action = int(torch.argmax(masked_logits, dim=1).item())

            actions = [0, 0, 0, 0]
            actions[controlled_id] = action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

        alive = [int(p[2]) for p in obs["players"]]
        my_alive = alive[controlled_id]
        alive_count = sum(alive)
        if my_alive == 1 and alive_count == 1:
            wins += 1
        elif my_alive == 1:
            draws += 1
        else:
            losses += 1

    print(f"Quick eval proxy | wins={wins} draws={draws} losses={losses}", flush=True)


# =============================================================================
# Main
# =============================================================================
def main():
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    if BASELINE_IMPORT_ERRORS:
        print("Some baseline imports failed; trainer will use fallback rules where needed.", flush=True)
        for name, err in BASELINE_IMPORT_ERRORS:
            print(f"  - {name}: {err}", flush=True)

    print("=== Phase 1: Collect initial demonstrations ===", flush=True)
    collect_initial_data(TRAIN_DIR, VAL_DIR, INITIAL_GAMES)

    print("=== Phase 2: Train initial policy ===", flush=True)
    model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=None, lr=LEARNING_RATE)

    for round_idx in range(DAGGER_ROUNDS):
        print(f"=== Phase 3.{round_idx + 1}: DAgger collection ===", flush=True)
        new_samples = collect_dagger_data(model, TRAIN_DIR, DAGGER_GAMES_PER_ROUND)
        print(f"DAgger round {round_idx + 1}: collected {new_samples} correction samples", flush=True)

        print(f"=== Phase 4.{round_idx + 1}: Fine-tune with aggregated data ===", flush=True)
        model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_PATH, lr=FINE_TUNE_LR)

    print("=== Optional quick sanity check ===", flush=True)
    quick_eval_against_baselines(model, num_games=20)
    print("Done.", flush=True)




# =============================================================================
# Best-practice fine-tuning additions
# =============================================================================
import copy
from torch.distributions import Categorical

# Training knobs for the actor-critic stage.
RL_ROUNDS = 3
ROLLOUT_GAMES_PER_ROUND = 120
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 256
PPO_CLIP_EPS = 0.20
PPO_GAMMA = 0.98
PPO_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 1.0
BC_MIX_COEF = 0.15
MIXED_DAGGER_GAMES = 120


RL_ROUNDS = 4
ROLLOUT_GAMES_PER_ROUND = 250
PPO_EPOCHS = 6
PPO_BATCH_SIZE = 128
PPO_CLIP_EPS = 0.15
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.02
PPO_MAX_GRAD_NORM = 1
BC_MIX_COEF = 0.10


# -----------------------------------------------------------------------------
# Policy/value model
# -----------------------------------------------------------------------------
class BomberNet(nn.Module):
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_actions: int = NUM_ACTIONS, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.10),
            ResidualBlock(width, dropout=0.10),
            ResidualBlock(width, dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(64, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.stem(x)
        feat = self.blocks(feat)
        feat = self.pool(feat)
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value


def _model_logits_value(model: nn.Module, states: torch.Tensor):
    out = model(states)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, None


def _policy_logits(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    logits, _ = _model_logits_value(model, states)
    return logits


def _legal_action_mask(grid: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int], bombs_left: int) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in legal_actions(grid, bombs, my_pos, bombs_left):
        mask[int(a)] = 1.0
    if mask.sum() <= 0:
        mask[0] = 1.0
    return mask




def _shielded_legal_mask(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    my_id: int,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """
    Safety shield for rollout collection.

    This is applied BEFORE action sampling so the sampled action and its log_prob
    still match the executed behavior policy.
    """
    mask = np.array(legal_mask, dtype=np.float32, copy=True)

    if my_id >= len(players) or int(players[my_id][2]) != 1:
        if mask.sum() <= 0:
            mask[0] = 1.0
        return mask

    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    blocked = bomb_positions_set(bombs)

    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
    danger_soon = danger_plane(grid, players, bombs, timer_threshold=2)
    in_danger = bool(danger_now[my_pos[0], my_pos[1]] > 0.0 or danger_soon[my_pos[0], my_pos[1]] > 0.0)

    # If we're already threatened, keep only actions that move to a tile
    # with a real time-safe escape margin.
    if in_danger:
        safe_moves = []
        for a in (1, 2, 3, 4):
            if mask[a] <= 0.0:
                continue
            nr, nc = next_pos(my_pos, a)
            if not passable(grid, nr, nc):
                mask[a] = 0.0
                continue
            if (nr, nc) in blocked:
                mask[a] = 0.0
                continue
            if escape_margin_from_position(grid, players, bombs, (nr, nc), max_depth=6) > 0.0:
                safe_moves.append(a)
            else:
                mask[a] = 0.0

        # Prefer escaping over waiting if any escape exists.
        if safe_moves:
            mask[0] = 0.0
        else:
            # Keep STOP as the least-bad fallback if no escape is proven.
            if mask[0] <= 0.0:
                mask[0] = 1.0

    else:
        # Outside immediate danger, suppress suicidal bomb placements.
        if mask[5] > 0.0 and not should_place_bomb_here(grid, players, bombs, my_id, my_pos):
            mask[5] = 0.0

    if mask.sum() <= 0.0:
        mask[0] = 1.0

    return mask


def _sample_masked_action(
    model: nn.Module,
    state: torch.Tensor,
    legal_mask: np.ndarray,
    sample: bool = True,
    temperature: float = 1.0,
):
    logits, value = _model_logits_value(model, state)
    logits = logits / max(float(temperature), 1e-6)

    mask = torch.tensor(legal_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9

    dist = Categorical(logits=masked_logits)
    if sample:
        action = dist.sample()
    else:
        action = torch.argmax(masked_logits, dim=-1)

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return int(action.item()), float(log_prob.item()), float(entropy.item()), float(value.item()) if value is not None else 0.0


# -----------------------------------------------------------------------------
# Stronger opponent wrappers
# -----------------------------------------------------------------------------
class FrozenPolicyAgent:
    def __init__(self, agent_id: int, model: nn.Module, deterministic: bool = True):
        self.agent_id = int(agent_id)
        self.model = model
        self.deterministic = bool(deterministic)

    def act(self, obs: Dict) -> int:
        if self.agent_id >= len(obs["players"]) or int(obs["players"][self.agent_id][2]) != 1:
            return 0
        step = 0
        state = encode_obs(obs["map"], obs["players"], obs["bombs"], self.agent_id, step).unsqueeze(0).to(DEVICE)
        my_pos = (int(obs["players"][self.agent_id][0]), int(obs["players"][self.agent_id][1]))
        bombs_left = int(obs["players"][self.agent_id][3])
        legal_mask = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bombs_left)
        shielded_mask = _shielded_legal_mask(
            obs["map"],
            obs["players"],
            obs["bombs"],
            self.agent_id,
            legal_mask,
        )

        with torch.no_grad():
            action, _, _, _ = _sample_masked_action(
                self.model,
                state,
                legal_mask=shielded_mask,
                sample=not self.deterministic,
                temperature=1.0,
            )
        return int(action)


def build_selfplay_opponents(controlled_id: int, game_seed: int, frozen_model: Optional[nn.Module] = None) -> Dict[int, object]:
    """
    Mix baselines with one frozen-policy opponent when available.
    This gives the learner a moving target without making rollout generation unstable.
    """
    rng = random.Random(game_seed)
    other_ids = [pid for pid in range(4) if pid != controlled_id]

    baseline_pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent, BoxFarmerAgent, SimpleRuleAgent] if cls is not None]
    if not baseline_pool:
        baseline_pool = [_FallbackRuleAgent]

    opponents = {}
    chosen = []

    # With decent probability, include a frozen self-play copy for one slot.
    if frozen_model is not None and rng.random() < 0.80:
        self_play_slot = rng.choice(other_ids)
        opponents[self_play_slot] = FrozenPolicyAgent(self_play_slot, frozen_model, deterministic=True)
        chosen.append(self_play_slot)

    remaining_ids = [pid for pid in other_ids if pid not in chosen]
    for pid in remaining_ids:
        cls = rng.choice(baseline_pool)
        opponents[pid] = cls(pid)

    return opponents


# -----------------------------------------------------------------------------
# Reward shaping
# -----------------------------------------------------------------------------
def compute_shaped_reward(
    prev_obs: Dict,
    next_obs: Dict,
    my_id: int,
    action: int,
    terminated: bool,
    truncated: bool,
) -> float:
    reward = 0.02  # survive one more step

    prev_players = prev_obs["players"]
    next_players = next_obs["players"]

    if my_id < len(prev_players) and my_id < len(next_players):
        prev_alive = int(prev_players[my_id][2])
        next_alive = int(next_players[my_id][2])

        if prev_alive == 1 and next_alive == 0:
            reward -= 3.0

        prev_bonus = int(prev_players[my_id][4])
        next_bonus = int(next_players[my_id][4])
        reward += 0.12 * max(0, next_bonus - prev_bonus)

        prev_bombs_left = int(prev_players[my_id][3])
        next_bombs_left = int(next_players[my_id][3])
        reward += 0.08 * max(0, next_bombs_left - prev_bombs_left)

    prev_enemy_alive = int(np.sum(prev_players[:, 2])) if len(prev_players) else 0
    next_enemy_alive = int(np.sum(next_players[:, 2])) if len(next_players) else 0
    if my_id < len(prev_players):
        prev_enemy_alive -= int(prev_players[my_id][2])
    if my_id < len(next_players):
        next_enemy_alive -= int(next_players[my_id][2])
    reward += 1.0 * max(0, prev_enemy_alive - next_enemy_alive)

    prev_boxes = int(np.sum(prev_obs["map"] == 2))
    next_boxes = int(np.sum(next_obs["map"] == 2))
    reward += 0.05 * max(0, prev_boxes - next_boxes)

    if action == 5:
        reward -= 0.01

    if terminated or truncated:
        if my_id < len(next_players) and int(next_players[my_id][2]) == 1:
            alive_count = int(np.sum(next_players[:, 2]))
            if alive_count == 1:
                reward += 4.0
            else:
                reward += 1.0

    return float(np.clip(reward, -5.0, 5.0))


# -----------------------------------------------------------------------------
# Rollout storage
# -----------------------------------------------------------------------------
@dataclass
class RolloutEpisode:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]
    masks: List[np.ndarray]


def _discounted_returns(rewards: List[float], dones: List[bool], gamma: float = PPO_GAMMA) -> np.ndarray:
    returns = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _flatten_episodes(episodes: List[RolloutEpisode]):
    states, actions, old_log_probs, values, returns, masks = [], [], [], [], [], []
    advs = []
    for ep in episodes:
        if not ep.states:
            continue
        ret = _discounted_returns(ep.rewards, ep.dones)
        val = np.asarray(ep.values, dtype=np.float32)
        adv = ret - val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv
        states.extend(ep.states)
        actions.extend(ep.actions)
        old_log_probs.extend(ep.log_probs)
        values.extend(ep.values)
        returns.extend(ret.tolist())
        masks.extend(ep.masks)
        advs.extend(adv.tolist())

    if not states:
        raise RuntimeError("No rollout samples collected.")
    return (
        torch.tensor(np.asarray(states), dtype=torch.float32),
        torch.tensor(np.asarray(actions), dtype=torch.long),
        torch.tensor(np.asarray(old_log_probs), dtype=torch.float32),
        torch.tensor(np.asarray(values), dtype=torch.float32),
        torch.tensor(np.asarray(returns), dtype=torch.float32),
        torch.tensor(np.asarray(advs), dtype=torch.float32),
        torch.tensor(np.asarray(masks), dtype=torch.float32),
    )


# -----------------------------------------------------------------------------
# BC collection + training overrides
# -----------------------------------------------------------------------------
def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int) -> int:
    """
    DAgger collection using the current policy as the student and the teacher ensemble as the oracle.
    """
    ensure_dir(out_dir)
    model.eval()

    out_manifest = load_manifest(out_dir)
    chunk_idx = len(out_manifest.get("chunks", []))
    buf_states, buf_actions, buf_seeds = [], [], []
    collected = 0

    def flush_buffer():
        nonlocal chunk_idx, collected
        if not buf_states:
            return
        entry = flush_chunk(out_dir, chunk_idx, buf_states, buf_actions, buf_seeds)
        if entry:
            out_manifest["chunks"].append(entry)
            save_manifest(out_dir, out_manifest)
            collected += entry["count"]
            chunk_idx += 1
        buf_states.clear()
        buf_actions.clear()
        buf_seeds.clear()

    for game_idx in range(num_games):
        seed = 100000 + SEED + game_idx
        controlled_id = game_idx % 4

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherEnsemble(controlled_id)
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = _model_logits_value(model, state)
                student_action = int(torch.argmax(logits, dim=1).item())

            expert_action = int(teacher.act(obs))

            if student_action != expert_action or (student_action == 0 and expert_action != 0):
                buf_states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
                buf_actions.append(expert_action)
                buf_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = student_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            if len(buf_states) >= CHUNK_SIZE:
                flush_buffer()

        if (game_idx + 1) % 50 == 0:
            print(f"DAgger {game_idx + 1}/{num_games} games | new samples={collected + len(buf_actions)}", flush=True)

    flush_buffer()
    return collected


class ChunkedBCDataset(IterableDataset):
    def __init__(self, chunk_dir: str, augment: bool, shuffle_chunks: bool, shuffle_within_chunk: bool, seed: int):
        super().__init__()
        self.chunk_dir = chunk_dir
        self.augment = augment
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed = seed
        self.manifest = load_manifest(chunk_dir)
        self.chunks = list(self.manifest.get("chunks", []))
        self.total_len = int(sum(int(c.get("count", 0)) for c in self.chunks))

    def __len__(self) -> int:
        return self.total_len

    def __iter__(self):
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers

        rng = np.random.default_rng(self.seed + worker_id * 1337)
        chunk_indices = np.arange(len(self.chunks))
        if self.shuffle_chunks:
            rng.shuffle(chunk_indices)
        chunk_indices = chunk_indices[worker_id::num_workers]

        for chunk_idx in chunk_indices:
            meta = self.chunks[int(chunk_idx)]
            data = np.load(os.path.join(self.chunk_dir, meta["file"]))
            states = data["states"]
            actions = data["actions"]

            idxs = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(idxs)

            for i in idxs:
                state = torch.from_numpy(states[int(i)]).float()
                action = int(actions[int(i)])
                if self.augment:
                    state, action = augment_tensor_and_action(state, action)
                yield state, torch.tensor(action, dtype=torch.long)


def build_loaders(train_dir: str, val_dir: str):
    train_ds = ChunkedBCDataset(
        train_dir,
        augment=True,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        seed=SEED,
    )
    val_ds = ChunkedBCDataset(
        val_dir,
        augment=False,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        seed=SEED,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation samples found in {val_dir}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    class_weights = compute_class_weights(train_dir).to(DEVICE)
    return train_loader, val_loader, class_weights


def _run_bc_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (states, actions) in enumerate(loader):
        states = states.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits, _ = _model_logits_value(model, states)
        loss = criterion(logits, actions)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        total_loss += float(loss.item()) * states.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == actions).sum().item())
        total_count += int(states.size(0))

        if batch_idx % 50 == 0:
            print(f"  batch={batch_idx}", flush=True)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def train_policy_model(train_dir: str, val_dir: str, init_model_path: Optional[str] = None, lr: float = LEARNING_RATE):
    train_loader, val_loader, class_weights = build_loaders(train_dir, val_dir)

    model = BomberNet(INPUT_CHANNELS).to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        state = torch.load(init_model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_acc = _run_bc_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = _run_bc_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_left = PATIENCE
            print(f"  -> saved best model to {BEST_MODEL_PATH}", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  -> early stopping", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}", flush=True)
    return model


# -----------------------------------------------------------------------------
# Self-play rollouts + PPO fine-tuning
# -----------------------------------------------------------------------------
def collect_selfplay_rollouts(model: nn.Module, frozen_model: Optional[nn.Module], num_games: int) -> List[RolloutEpisode]:
    model.eval()
    if frozen_model is not None:
        frozen_model.eval()

    episodes: List[RolloutEpisode] = []
    for game_idx in range(num_games):
        seed = 300000 + SEED + game_idx
        controlled_id = game_idx % 4

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_selfplay_opponents(controlled_id, seed, frozen_model=frozen_model)

        ep = RolloutEpisode(states=[], actions=[], rewards=[], dones=[], log_probs=[], values=[], masks=[])
        done = False
        step = 0

        while not done:
            if controlled_id >= len(obs["players"]) or int(obs["players"][controlled_id][2]) != 1:
                break

            state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][controlled_id][0]), int(obs["players"][controlled_id][1]))
            bombs_left = int(obs["players"][controlled_id][3])
            legal_mask = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bombs_left)
            shielded_mask = _shielded_legal_mask(
                obs["map"],
                obs["players"],
                obs["bombs"],
                controlled_id,
                legal_mask,
            )

            with torch.no_grad():
                action, log_prob, _, value = _sample_masked_action(
                    model,
                    state,
                    legal_mask=shielded_mask,
                    sample=True,
                    temperature=0.90,
                )

            actions = [0, 0, 0, 0]
            actions[controlled_id] = int(action)
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            prev_obs = obs
            next_obs, terminated, truncated = env.step(actions)
            reward = compute_shaped_reward(prev_obs, next_obs, controlled_id, action, terminated, truncated)

            ep.states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
            ep.actions.append(int(action))
            ep.rewards.append(float(reward))
            ep.dones.append(bool(terminated or truncated or int(next_obs["players"][controlled_id][2]) == 0))
            ep.log_probs.append(float(log_prob))
            ep.values.append(float(value))
            ep.masks.append(shielded_mask.astype(np.float32))

            obs = next_obs
            done = bool(terminated or truncated or int(next_obs["players"][controlled_id][2]) == 0)
            step += 1

        if ep.states:
            episodes.append(ep)

        if (game_idx + 1) % 25 == 0:
            total_steps = sum(len(e.states) for e in episodes)
            print(f"Rollouts {game_idx + 1}/{num_games} games | episodes={len(episodes)} | steps={total_steps}", flush=True)

    return episodes


def ppo_finetune(model: nn.Module, episodes: List[RolloutEpisode], bc_mix_dir: Optional[str] = None):
    if not episodes:
        return model

    states, actions, old_log_probs, old_values, returns, advantages, masks = _flatten_episodes(episodes)

    dataset_size = states.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)

    # Optional BC regularizer: keep some imitation pressure so the policy does not drift too far.
    bc_loader = None
    if bc_mix_dir is not None and os.path.exists(bc_mix_dir):
        try:
            bc_ds = ChunkedBCDataset(
                bc_mix_dir,
                augment=True,
                shuffle_chunks=True,
                shuffle_within_chunk=True,
                seed=SEED + 999,
            )
            if len(bc_ds) > 0:
                bc_loader = DataLoader(
                    bc_ds,
                    batch_size=min(128, BATCH_SIZE),
                    shuffle=False,
                    num_workers=0,
                    drop_last=True,
                )
        except Exception:
            bc_loader = None

    model.train(True)
    idxs = np.arange(dataset_size)

    bc_iter = iter(bc_loader) if bc_loader is not None else None

    for epoch in range(1, PPO_EPOCHS + 1):
        np.random.shuffle(idxs)
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        batches = 0

        for start in range(0, dataset_size, PPO_BATCH_SIZE):
            batch_idx = idxs[start:start + PPO_BATCH_SIZE]
            if len(batch_idx) == 0:
                continue

            b_states = states[batch_idx].to(DEVICE)
            b_actions = actions[batch_idx].to(DEVICE)
            b_old_log_probs = old_log_probs[batch_idx].to(DEVICE)
            b_returns = returns[batch_idx].to(DEVICE)
            b_advantages = advantages[batch_idx].to(DEVICE)
            b_masks = masks[batch_idx].to(DEVICE)

            logits, values = _model_logits_value(model, b_states)
            masked_logits = logits.clone()
            masked_logits[b_masks <= 0.0] = -1e9

            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - b_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)
            policy_loss = -torch.mean(torch.min(ratio * b_advantages, clipped_ratio * b_advantages))
            value_loss = torch.mean((values - b_returns) ** 2)

            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            # Small BC anchor if a replay loader is available.
            if bc_iter is not None:
                try:
                    bc_states, bc_actions = next(bc_iter)
                except StopIteration:
                    bc_iter = iter(bc_loader)
                    bc_states, bc_actions = next(bc_iter)
                bc_states = bc_states.to(DEVICE)
                bc_actions = bc_actions.to(DEVICE)
                bc_logits, _ = _model_logits_value(model, bc_states)
                bc_loss = nn.functional.cross_entropy(bc_logits, bc_actions)
                loss = loss + BC_MIX_COEF * bc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
            optimizer.step()

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_loss += float(loss.item())
            batches += 1

        print(
            f"PPO epoch {epoch:02d}/{PPO_EPOCHS} | "
            f"loss={total_loss / max(1, batches):.4f} | "
            f"policy={total_policy / max(1, batches):.4f} | "
            f"value={total_value / max(1, batches):.4f} | "
            f"entropy={total_entropy / max(1, batches):.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), MODEL_PATH)
    return model


# -----------------------------------------------------------------------------
# Evaluation override
# -----------------------------------------------------------------------------
def quick_eval_against_baselines(model: nn.Module, num_games: int = 20) -> None:
    model.eval()
    wins = 0
    draws = 0
    losses = 0
    total_survival_steps = 0

    for game_idx in range(num_games):
        seed = 400000 + SEED + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            if controlled_id >= len(obs["players"]) or int(obs["players"][controlled_id][2]) != 1:
                break
            state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][controlled_id][0]), int(obs["players"][controlled_id][1]))
            bombs_left = int(obs["players"][controlled_id][3])
            legal_mask = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bombs_left)
            with torch.no_grad():
                action, _, _, _ = _sample_masked_action(model, state, legal_mask=legal_mask, sample=False)
            actions = [0, 0, 0, 0]
            actions[controlled_id] = action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))
            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

        total_survival_steps += step
        alive = [int(p[2]) for p in obs["players"]]
        my_alive = alive[controlled_id]
        alive_count = sum(alive)
        if my_alive == 1 and alive_count == 1:
            wins += 1
        elif my_alive == 1:
            draws += 1
        else:
            losses += 1

    avg_steps = total_survival_steps / max(1, num_games)
    print(f"Quick eval proxy | wins={wins} draws={draws} losses={losses} | avg_survival_steps={avg_steps:.1f}", flush=True)


# -----------------------------------------------------------------------------
# Main override
# -----------------------------------------------------------------------------
def main():
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    # if BASELINE_IMPORT_ERRORS:
    #     print("Some baseline imports failed; trainer will use fallback rules where needed.", flush=True)
    #     for name, err in BASELINE_IMPORT_ERRORS:
    #         print(f"  - {name}: {err}", flush=True)

    # print("=== Phase 1: Collect initial demonstrations ===", flush=True)
    # collect_initial_data(TRAIN_DIR, VAL_DIR, INITIAL_GAMES)

    # print("=== Phase 2: Train BC policy/value backbone ===", flush=True)
    # model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=None, lr=LEARNING_RATE)

    # print("=== Phase 3: DAgger correction pass ===", flush=True)
    # new_samples = collect_dagger_data(model, TRAIN_DIR, MIXED_DAGGER_GAMES)
    # print(f"DAgger collected {new_samples} corrective samples", flush=True)

    # print("=== Phase 4: Refresh BC after DAgger ===", flush=True)
    # model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_PATH, lr=FINE_TUNE_LR)
    
    model = BomberNet(INPUT_CHANNELS).to(DEVICE)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "model_bc_best.pth")
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        raise FileNotFoundError(f"Can't find {pretrained_path}")


    print("=== Phase 5: Self-play actor-critic fine-tuning ===", flush=True)
    for round_idx in range(RL_ROUNDS):
        frozen = copy.deepcopy(model).to(DEVICE)
        frozen.eval()
        rollouts = collect_selfplay_rollouts(model, frozen_model=frozen, num_games=ROLLOUT_GAMES_PER_ROUND)
        print(f"Round {round_idx + 1}: collected {len(rollouts)} episodes", flush=True)
        model = ppo_finetune(model, rollouts, bc_mix_dir=TRAIN_DIR)

    print("=== Optional quick sanity check ===", flush=True)
    quick_eval_against_baselines(model, num_games=20)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
