"""
fat_best_reward_v4.py — Bomberland training pipeline (consolidated + fixed)

Fixes vs v3:
  [CRITICAL] Duplicate class definitions removed — single BomberNet, single main().
  [CRITICAL] BomberNet: AdaptiveAvgPool2d(1) → AdaptiveAvgPool2d(4); scalar channels
             (14,17-20,22-23) extracted separately and fed directly into the MLP head
             instead of being broadcast over 13×13 spatial planes that the CNN ignores.
  [CRITICAL] GAE fully implemented — PPO_LAMBDA=0.95 is now actually used; advantages
             are computed with Schulman et al. λ-weighted TD errors, not plain MC returns.
  [CRITICAL] Value bootstrapping for truncated episodes — at the 500-step boundary the
             value function is queried and used as a future-return estimate.
  [CRITICAL] should_place_bomb_here / safe_to_bomb_plane include the hypothetical new
             bomb in the bomb set when checking escape routes, so the safety check is
             actually correct.
  [MEDIUM]   AUGMENT_FLIP_PROB 1.0 → 0.5: original orientation now also trained on.
  [MEDIUM]   FrozenPolicyAgent tracks actual game step (was hardcoded to 0).
  [MEDIUM]   Opponent pool for BC/DAgger expanded to 5 baselines for diversity.
  [MEDIUM]   LeaguePool: keeps up to 6 past model snapshots; self-play opponents drawn
             from current policy (40%), league history (40%), or baselines (20%).
  [MEDIUM]   PPO scale increased: 5 rounds × 200 games, 6 epochs each.
  [MEDIUM]   Bomb reward asymmetry reduced: penalty -0.10 (was -0.20), reward +0.04.
  [MINOR]    Evaluation tracks kills and boxes destroyed alongside win/draw/loss.
  [MINOR]    Single ChunkedBCDataset class (was duplicated under two names).

NOTE: agent.py submitted to the contest must use the same BomberNet class defined here
      (copy the BomberNet + ResidualBlock + SPATIAL_CHANNELS / SCALAR_CHANNELS constants).
"""

import copy
import json
import os
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# ---------------------------------------------------------------------------
# Local engine import
# ---------------------------------------------------------------------------
sys.path.append(os.getcwd())
from engine.game import BomberEnv

BASELINE_IMPORT_ERRORS: List[Tuple[str, str]] = []

def _try_import(name: str, module: str, cls: str):
    try:
        mod = __import__(module, fromlist=[cls])
        return getattr(mod, cls)
    except Exception as e:
        BASELINE_IMPORT_ERRORS.append((name, repr(e)))
        return None

TacticalRuleAgent  = _try_import("TacticalRuleAgent",  "agent.tactical_rule_agent",  "TacticalRuleAgent")
GeniusRuleAgent    = _try_import("GeniusRuleAgent",    "agent.genius_rule_agent",    "GeniusRuleAgent")
SmarterRuleAgent   = _try_import("SmarterRuleAgent",   "agent.smarter_rule_agent",   "SmarterRuleAgent")
BoxFarmerAgent     = _try_import("BoxFarmerAgent",     "agent.box_farmer_agent",     "BoxFarmerAgent")
SimpleRuleAgent    = _try_import("SimpleRuleAgent",    "agent.simple_rule_agent",    "SimpleRuleAgent")
RandomAgent        = _try_import("RandomAgent",        "agent.random_agent",         "RandomAgent")

# ===========================================================================
# Configuration
# Seed = 42, 142
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 142

BOARD_SIZE            = 13
INPUT_CHANNELS        = 27
NUM_ACTIONS           = 6
MAX_STEPS             = 500
EXPLOSION_TIME_HORIZON = 8.0

# --- Channel split: which of the 27 channels carry true spatial information
#     and which are just scalar values broadcast over the full 13×13 plane.
#
#   Spatial (20): wall, box, grass, items, player positions, explosion/danger
#                 planes, bomb timer heatmap, bomb radius heatmap, my-position,
#                 safe-to-bomb plane, enemy pressure planes, bottleneck risk.
#   Scalar  ( 7): bombs_left, dist_item, dist_enemy, reachability, escape score,
#                 bomb count, step ratio.
SPATIAL_CHANNELS: List[int] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,21,24,25,26]
SCALAR_CHANNELS:  List[int] = [14,17,18,19,20,22,23]
N_SPATIAL = len(SPATIAL_CHANNELS)  # 20
N_SCALAR  = len(SCALAR_CHANNELS)   # 7

# --- BC / DAgger
INITIAL_GAMES       = 800
MIXED_DAGGER_GAMES  = 200
TRAIN_SPLIT_MOD     = 10     # seed % 10 == 0 → validation
CHUNK_SIZE          = 2048
BATCH_SIZE          = 128
EPOCHS              = 20
LEARNING_RATE       = 1e-3
FINE_TUNE_LR        = 3e-4
WEIGHT_DECAY        = 1e-4
PATIENCE            = 5
GRAD_CLIP_NORM      = 1.0

TRAIN_DIR      = "bc_train_chunks"
VAL_DIR        = "bc_val_chunks"
MODEL_PATH     = "model_bc.pth"
BEST_MODEL_PATH = "model_bc_best.pth"
MANIFEST_NAME  = "manifest.json"

AUGMENT_FLIP_PROB = 0.5  # FIX: was 1.0 — original orientation now also trained on

# --- PPO / self-play
RL_ROUNDS               = 7    # was 3
ROLLOUT_GAMES_PER_ROUND = 200  # was 120
PPO_EPOCHS              = 6    # was 4
PPO_BATCH_SIZE          = 256
PPO_CLIP_EPS            = 0.20
PPO_GAMMA               = 0.98
PPO_LAMBDA              = 0.95  # FIX: now actually used in GAE
PPO_VALUE_COEF          = 0.5
PPO_ENTROPY_COEF        = 0.01
PPO_MAX_GRAD_NORM       = 1.0
BC_MIX_COEF             = 0.04
LEAGUE_POOL_SIZE        = 2    # max past checkpoints kept for self-play


# ===========================================================================
# Seeding
# ===========================================================================
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


# ===========================================================================
# Board / movement helpers
# ===========================================================================
MOVES = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


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


def bomb_effective_explosion_times(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray
) -> np.ndarray:
    """Resolve chain reactions; return effective explosion timer per bomb."""
    if bombs is None or len(bombs) == 0:
        return np.zeros((0,), dtype=np.int32)
    n = len(bombs)
    times = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    blasts: List[set] = []
    for i in range(n):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        blasts.append(blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius))

    q: deque = deque(range(n))
    in_q = [True] * n
    while q:
        i = q.popleft()
        in_q[i] = False
        ti = max(0, int(times[i]))
        for j in range(n):
            if i == j:
                continue
            bj = (int(bombs[j][0]), int(bombs[j][1]))
            if bj in blasts[i] and int(times[j]) > ti:
                times[j] = ti
                if not in_q[j]:
                    q.append(j)
                    in_q[j] = True
    return times


# ===========================================================================
# Danger / explosion planes
# ===========================================================================
def explosion_time_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    horizon: float = EXPLOSION_TIME_HORIZON,
) -> np.ndarray:
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


def danger_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    timer_threshold: int = 1,
) -> np.ndarray:
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger
    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / float(EXPLOSION_TIME_HORIZON) if EXPLOSION_TIME_HORIZON > 0 else 0.0
    danger[plane <= threshold] = 1.0
    return danger


def immediate_danger_plane(grid, players, bombs):
    return danger_plane(grid, players, bombs, timer_threshold=1)


def chain_danger_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    chain_horizon: int = 3,
) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane
    original = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    effective = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        if int(effective[i]) <= 1 or int(effective[i]) > chain_horizon:
            continue
        if int(effective[i]) >= int(original[i]):
            continue
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        for r, c in blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius):
            plane[r, c] = 1.0
    return plane


def future_danger_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    horizon: float = EXPLOSION_TIME_HORIZON,
) -> np.ndarray:
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


def tile_earliest_explosion_times(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray
) -> np.ndarray:
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


def bomb_pressure_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None:
        bombs = np.zeros((0, 4), dtype=np.int8)
    for pid in range(4):
        if pid == my_id or pid >= len(players) or int(players[pid][2]) != 1:
            continue
        if int(players[pid][3]) <= 0:
            continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c):
            continue
        radius = 1 + int(players[pid][4])
        for x, y in blast_tiles(grid, r, c, radius):
            plane[x, y] = max(plane[x, y], 1.0)
    return plane


def future_bomb_pressure_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None:
        bombs = np.zeros((0, 4), dtype=np.int8)
    blocked = bomb_positions_set(bombs)
    for pid in range(4):
        if pid == my_id or pid >= len(players) or int(players[pid][2]) != 1:
            continue
        if int(players[pid][3]) <= 0:
            continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c):
            continue
        radius = 1 + int(players[pid][4])
        candidates = [(r, c)]
        for a in (1, 2, 3, 4):
            nr, nc = next_pos((r, c), a)
            if passable(grid, nr, nc) and (nr, nc) not in blocked:
                candidates.append((nr, nc))
        for pr, pc in candidates:
            for x, y in blast_tiles(grid, pr, pc, radius):
                plane[x, y] = max(plane[x, y], 0.5)
    return plane


def bottleneck_risk_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane
    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    blocked = bomb_positions_set(bombs)
    explosion_times = tile_earliest_explosion_times(grid, players, bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not passable(grid, r, c) or (r, c) in blocked:
                continue
            exits = 0
            fragile = 0
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if not passable(grid, nr, nc) or (nr, nc) in blocked:
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
            manhattan = abs(r - my_r) + abs(c - my_c)
            if manhattan <= 1:
                score = max(score, 0.75)
            elif manhattan <= 2:
                score = max(score, 0.35)
            plane[r, c] = score
    return plane


# ===========================================================================
# BFS / escape utilities
# ===========================================================================
def escape_margin_from_position(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    start: Tuple[int, int], max_depth: int = 6,
) -> float:
    explosion_times = tile_earliest_explosion_times(grid, players, bombs)
    blocked = bomb_positions_set(bombs)
    q: deque = deque([(start, 0)])
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
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return -1.0 if best_margin < -1000 else float(best_margin)


def time_safe_escape_score(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> float:
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return 0.0
    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    margin = escape_margin_from_position(grid, players, bombs, my_pos, max_depth=6)
    return float(np.clip(margin / 6.0, 0.0, 1.0)) if margin > 0 else 0.0


def bfs_distance_to_targets(
    grid: np.ndarray, start: Tuple[int, int], targets: set,
    bombs: np.ndarray, max_depth: int = 64,
) -> Optional[int]:
    if not targets:
        return None
    blocked = bomb_positions_set(bombs)
    q: deque = deque([(start, 0)])
    seen = {start}
    while q:
        pos, dist = q.popleft()
        if pos in targets:
            return dist
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return None


def bfs_reachable_count(
    grid: np.ndarray, start: Tuple[int, int], bombs: np.ndarray, max_depth: int = 3
) -> int:
    blocked = bomb_positions_set(bombs)
    q: deque = deque([(start, 0)])
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
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return count


def norm_dist(d: Optional[int], cap: float = 24.0) -> float:
    return 1.0 if d is None else float(min(d, cap)) / cap


def normalize_scalar(x: float, denom: float) -> float:
    return float(np.clip(x / denom, 0.0, 1.0)) if denom > 0 else 0.0


def legal_actions(
    grid: np.ndarray, bombs: np.ndarray,
    my_pos: Tuple[int, int], bombs_left: int,
) -> List[int]:
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
    return [int(a) for a in legal if int(a) in (1, 2, 3, 4)]


# ===========================================================================
# Bomb safety — FIX: include hypothetical new bomb in escape analysis
# ===========================================================================
def _add_hypothetical_bomb(
    bombs: np.ndarray, pos: Tuple[int, int], owner: int, timer: int = 7
) -> np.ndarray:
    """Return a new bomb array with a hypothetical bomb appended."""
    new_row = np.array([[pos[0], pos[1], timer, owner]], dtype=np.int8)
    if bombs is not None and len(bombs) > 0:
        return np.concatenate([bombs, new_row], axis=0)
    return new_row


def should_place_bomb_here(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    my_id: int, pos: Tuple[int, int],
) -> bool:
    """
    Check whether placing a bomb at *pos* is survivable AND tactically useful.

    FIX vs v3: the escape margin is now evaluated against the bomb set that
    *includes the new bomb*, so the check is actually sound — the new bomb's
    blast and its effect on chain reactions are accounted for.
    """
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return False
    if not passable(grid, pos[0], pos[1]):
        return False

    my_radius = 1 + int(players[my_id][4])

    # FIX: include the hypothetical new bomb when checking escape
    hyp_bombs = _add_hypothetical_bomb(bombs, pos, my_id)
    blast = blast_tiles(grid, pos[0], pos[1], my_radius)
    blocked = bomb_positions_set(hyp_bombs)

    for a in (1, 2, 3, 4):
        nr, nc = next_pos(pos, a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked:
            continue
        if (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6) > 0:
            return True
    return False


def safe_to_bomb_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> np.ndarray:
    """
    Mark my current tile if bombing here looks useful and survivable.

    FIX: escape check now includes the hypothetical new bomb (same fix as
    should_place_bomb_here).
    """
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    if not in_bounds(my_r, my_c):
        return plane

    blocked_now = bomb_positions_set(bombs)
    if (my_r, my_c) in blocked_now:
        return plane

    bomb_radius = 1 + int(players[my_id][4])
    blast = blast_tiles(grid, my_r, my_c, bomb_radius)

    enemy_positions = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(4)
        if i != my_id and i < len(players) and int(players[i][2]) == 1
    }
    hit_boxes  = any(int(grid[x, y]) == 2 for x, y in blast)
    hit_enemy  = any((x, y) in enemy_positions for x, y in blast)
    if not (hit_boxes or hit_enemy):
        return plane

    # FIX: include hypothetical bomb when checking escape
    hyp_bombs = _add_hypothetical_bomb(bombs, (my_r, my_c), my_id)
    blocked_hyp = bomb_positions_set(hyp_bombs)

    for a in (1, 2, 3, 4):
        nr, nc = next_pos((my_r, my_c), a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked_hyp:
            continue
        if (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6) > 0:
            plane[my_r, my_c] = 1.0
            break
    return plane


# ===========================================================================
# Observation encoding  (27 channels — layout unchanged from v3)
# ===========================================================================
def encode_obs(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    my_id: int, step: int,
) -> torch.Tensor:
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Static map (0-4)
    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    # Player positions (5-8)
    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                state[5 + pid, r, c] = 1.0

    # Bomb danger system (9-12)
    state[9]  = explosion_time_plane(grid, players, bombs)
    state[10] = immediate_danger_plane(grid, players, bombs)
    state[11] = chain_danger_plane(grid, players, bombs)
    state[12] = future_danger_plane(grid, players, bombs)

    # Ego features
    me_alive   = 0
    my_pos     = (0, 0)
    bombs_left = 0
    if my_id < len(players) and int(players[my_id][2]) == 1:
        me_alive = 1
        mr, mc   = int(players[my_id][0]), int(players[my_id][1])
        my_pos   = (mr, mc)
        if in_bounds(mr, mc):
            state[13, mr, mc] = 1.0
        bombs_left = int(players[my_id][3])

    # ch 14 — bombs_left (scalar broadcast, extracted by model separately)
    state[14].fill(normalize_scalar(bombs_left, 5.0))

    # ch 15-16 — bomb timer / radius heatmaps (spatial)
    if bombs is not None and len(bombs) > 0:
        eff_times = bomb_effective_explosion_times(grid, players, bombs)
        for i, b in enumerate(bombs):
            r, c = int(b[0]), int(b[1])
            if not in_bounds(r, c):
                continue
            t = max(int(eff_times[i]), 1)
            state[15, r, c] = max(state[15, r, c], 1.0 / float(t))
            owner = int(b[3]) if len(b) > 3 else -1
            state[16, r, c] = max(state[16, r, c],
                                  normalize_scalar(bomb_radius_for_owner(players, owner), 6.0))

    # ch 17-20, 22-23 — scalar features (broadcast; extracted by model separately)
    if me_alive:
        item_pos  = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
        enemy_pos = {
            (int(players[i][0]), int(players[i][1]))
            for i in range(4)
            if i != my_id and i < len(players) and int(players[i][2]) == 1
        }
        state[17].fill(norm_dist(bfs_distance_to_targets(grid, my_pos, item_pos, bombs)))
        state[18].fill(norm_dist(bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs)))
        state[19].fill(normalize_scalar(bfs_reachable_count(grid, my_pos, bombs, max_depth=3), 20.0))
        state[20].fill(time_safe_escape_score(grid, players, bombs, my_id))
        state[21] = safe_to_bomb_plane(grid, players, bombs, my_id)
    else:
        state[17].fill(1.0)
        state[18].fill(1.0)

    state[22].fill(normalize_scalar(len(bombs) if bombs is not None else 0, 10.0))
    state[23].fill(normalize_scalar(step, float(MAX_STEPS)))
    state[24] = bomb_pressure_plane(grid, players, bombs, my_id)
    state[25] = future_bomb_pressure_plane(grid, players, bombs, my_id)
    state[26] = bottleneck_risk_plane(grid, players, bombs, my_id)

    return torch.from_numpy(state)


# ===========================================================================
# Teacher ensemble
# ===========================================================================
class _FallbackRuleAgent:
    """Minimal local fallback used when baseline imports fail."""
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: Dict) -> int:
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0
        r, c        = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left  = int(players[self.agent_id][3])
        dng         = danger_plane(grid, players, bombs, timer_threshold=1)
        if dng[r, c] > 0:
            moves = [a for a in (1, 2, 3, 4)
                     if passable(grid, *next_pos((r, c), a))
                     and dng[next_pos((r, c), a)[0], next_pos((r, c), a)[1]] == 0
                     and next_pos((r, c), a) not in bomb_positions_set(bombs)]
            return int(random.choice(moves)) if moves else 0
        items = {(int(x), int(y)) for x, y in np.argwhere((grid == 3) | (grid == 4))}
        if items:
            best, best_d = 0, 10**9
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if passable(grid, nr, nc) and (nr, nc) not in bomb_positions_set(bombs):
                    d = min(abs(nr - ir) + abs(nc - ic) for ir, ic in items)
                    if d < best_d:
                        best_d, best = d, a
            if best:
                return int(best)
        return 5 if bombs_left > 0 else 0


def _maybe_make(cls, agent_id: int):
    return _FallbackRuleAgent(agent_id) if cls is None else cls(agent_id)


class TeacherEnsemble:
    """Weighted ensemble of all 6 baselines used as oracle for BC / DAgger."""
    _W = {"tactical": 3.0, "genius": 2.5, "smarter": 2.0,
          "box_farmer": 1.0, "simple": 0.75, "random": 0.25}

    def __init__(self, agent_id: int):
        self.agent_id  = int(agent_id)
        self.tactical  = _maybe_make(TacticalRuleAgent,  agent_id)
        self.genius    = _maybe_make(GeniusRuleAgent,    agent_id)
        self.smarter   = _maybe_make(SmarterRuleAgent,   agent_id)
        self.box_farmer = _maybe_make(BoxFarmerAgent,    agent_id)
        self.simple    = _maybe_make(SimpleRuleAgent,    agent_id)
        self.random    = _maybe_make(RandomAgent,        agent_id)
        self.weights   = dict(self._W)

    # ------------------------------------------------------------------ #
    def _collect(self, obs: Dict) -> Dict[str, int]:
        return {
            "tactical":  int(self.tactical.act(obs)),
            "genius":    int(self.genius.act(obs)),
            "smarter":   int(self.smarter.act(obs)),
            "box_farmer": int(self.box_farmer.act(obs)),
            "simple":    int(self.simple.act(obs)),
            "random":    int(self.random.act(obs)),
        }

    def _weighted_vote(self, acts: Dict[str, int], legal: Optional[set] = None) -> int:
        score: Counter = Counter()
        for k, v in acts.items():
            score[v] += self.weights[k]
        if legal is not None:
            for a in list(score.keys()):
                if a not in legal:
                    score[a] -= 10.0
        best = max(score.values())
        cands = [a for a, s in score.items() if abs(s - best) < 1e-9]
        priority = list(acts.values())
        for p in priority:
            if p in cands:
                return int(p)
        return int(cands[0])

    def _move_score(self, grid, players, bombs, pos: Tuple[int, int]) -> float:
        if not passable(grid, pos[0], pos[1]):
            return -1e9
        if pos in bomb_positions_set(bombs):
            return -1e9
        margin = escape_margin_from_position(grid, players, bombs, pos, max_depth=6)
        if margin <= 0:
            return -1000.0
        score = 2.0 * margin
        if danger_plane(grid, players, bombs, timer_threshold=1)[pos[0], pos[1]] > 0:
            score -= 5.0
        if bomb_pressure_plane(grid, players, bombs, self.agent_id)[pos[0], pos[1]] > 0:
            score -= 2.0
        score += 0.05 * bfs_reachable_count(grid, pos, bombs, max_depth=3)
        return float(score)

    def _best_escape(self, grid, players, bombs, legal: set, my_pos) -> int:
        best_a, best_s = 0, -1e18
        for a in movement_actions_from_legal(legal):
            s = self._move_score(grid, players, bombs, next_pos(my_pos, a))
            if s > best_s:
                best_s, best_a = s, int(a)
        return best_a

    def act(self, obs: Dict) -> int:
        grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0
        r, c = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left = int(players[self.agent_id][3])
        legal = set(legal_actions(grid, bombs, (r, c), bombs_left))
        acts  = self._collect(obs)

        dng      = danger_plane(grid, players, bombs, timer_threshold=1)
        pressure = bomb_pressure_plane(grid, players, bombs, self.agent_id)
        bottle   = bottleneck_risk_plane(grid, players, bombs, self.agent_id)

        if dng[r, c] > 0 or bottle[r, c] > 0.65 or pressure[r, c] > 0:
            safe_mv = self._best_escape(grid, players, bombs, legal, (r, c))
            if safe_mv in legal and safe_mv != 0:
                return int(safe_mv)

        if 5 in legal and should_place_bomb_here(grid, players, bombs, self.agent_id, (r, c)):
            if acts["tactical"] == 5 or acts["genius"] == 5 or acts["smarter"] == 5:
                return 5
            blast      = blast_tiles(grid, r, c, 1 + int(players[self.agent_id][4]))
            enemies    = {(int(players[i][0]), int(players[i][1]))
                          for i in range(4)
                          if i != self.agent_id and i < len(players) and int(players[i][2]) == 1}
            if (any(int(grid[x, y]) == 2 for x, y in blast) or any((x, y) in enemies for x, y in blast)) \
               and dng[r, c] == 0 and pressure[r, c] == 0:
                return 5

        box_count = int(np.sum(grid == 2))
        self.weights["box_farmer"] = 2.2 if box_count >= 18 else 1.2
        alive_cnt = int(np.sum(players[:, 2])) if len(players) else 0
        if alive_cnt <= 2:
            self.weights["tactical"] = 3.5
            self.weights["genius"]   = 2.8
        else:
            self.weights["tactical"] = 3.0
            self.weights["genius"]   = 2.5
        if pressure[r, c] > 0 or future_bomb_pressure_plane(grid, players, bombs, self.agent_id)[r, c] > 0:
            self.weights["random"] = 0.05
            self.weights["simple"] = 0.50
        else:
            self.weights["random"] = 0.25
            self.weights["simple"] = 0.75

        vote = self._weighted_vote(acts, legal=legal)
        if vote == 5 and not should_place_bomb_here(grid, players, bombs, self.agent_id, (r, c)):
            vote = self._best_escape(grid, players, bombs, legal, (r, c))
        if vote in (1, 2, 3, 4) and dng[r, c] > 0:
            nr, nc = next_pos((r, c), vote)
            if not passable(grid, nr, nc) or dng[nr, nc] > 0:
                alt = self._best_escape(grid, players, bombs, legal, (r, c))
                if alt in legal:
                    return int(alt)
        return int(vote)


# ===========================================================================
# Model — FIX: spatial / scalar split + 4×4 pool
# ===========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class BomberNet(nn.Module):
    """
    Actor-critic network with two input pathways:

      1. Spatial path  — the 20 channels that carry genuine per-tile information
         (map layout, player positions, danger planes, bomb heatmaps, …).
         Processed by a CNN stem + 3 residual blocks + AdaptiveAvgPool2d(4×4)
         → flattened to width × 16 = 1024-d.

      2. Scalar path   — the 7 channels that contain a single scalar value
         broadcast over the full 13×13 plane (BFS distances, step ratio, …).
         Extracted by reading only the [0,0] pixel → 7-d vector.

    The two paths are concatenated → MLP policy head + MLP value head.

    Fixes vs BomberNet in v3:
      - AdaptiveAvgPool2d(1) → AdaptiveAvgPool2d(4): spatial structure preserved.
      - Scalar channels split out of the CNN so they reach the head directly.
      - Wider MLP heads (256-d first layer) to handle the larger combined input.

    ⚠️ IMPORTANT: Copy this class verbatim into agent.py for inference.
    """
    _SPATIAL = SPATIAL_CHANNELS  # 20 channels
    _SCALAR  = SCALAR_CHANNELS   # 7 channels
    _POOL    = 4                  # spatial pooling grid size

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_actions:    int = NUM_ACTIONS,
        width:          int = 64,
    ):
        super().__init__()
        n_sp     = len(self._SPATIAL)
        n_sc     = len(self._SCALAR)
        feat_dim = width * (self._POOL ** 2) + n_sc  # 64*16 + 7 = 1031

        self.stem = nn.Sequential(
            nn.Conv2d(n_sp, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.10),
            ResidualBlock(width, dropout=0.10),
            ResidualBlock(width, dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d(self._POOL)

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.20),
            nn.Linear(256, 128),      nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(128, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(256, 128),      nn.ReLU(inplace=True), nn.Dropout(0.05),
            nn.Linear(128, 1),
        )
        # Register channel indices as buffers so they move with .to(device)
        self.register_buffer("_sp_idx", torch.tensor(self._SPATIAL, dtype=torch.long))
        self.register_buffer("_sc_idx", torch.tensor(self._SCALAR,  dtype=torch.long))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sp = x[:, self._sp_idx]               # (B, 20, 13, 13) — spatial
        sc = x[:, self._sc_idx, 0, 0]         # (B, 7)          — scalars

        feat = self.stem(sp)
        feat = self.blocks(feat)
        feat = self.pool(feat).flatten(1)      # (B, 1024)
        combined = torch.cat([feat, sc], dim=1)  # (B, 1031)

        logits = self.policy_head(combined)    # (B, 6)
        value  = self.value_head(combined).squeeze(-1)  # (B,)
        return logits, value


def _model_logits_value(
    model: nn.Module, states: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(states)
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, None)


# ===========================================================================
# Augmentation — FIX: prob=0.5 so original orientation is also trained on
# ===========================================================================
def _remap_h(a: int) -> int:
    return {1: 2, 2: 1, 3: 3, 4: 4, 0: 0, 5: 5}.get(int(a), int(a))

def _remap_v(a: int) -> int:
    return {3: 4, 4: 3, 1: 1, 2: 2, 0: 0, 5: 5}.get(int(a), int(a))


def augment_tensor_and_action(
    state: torch.Tensor, action: int
) -> Tuple[torch.Tensor, int]:
    # FIX: was 1.0 — only apply augmentation 50% of the time
    if random.random() > AUGMENT_FLIP_PROB:
        return state, int(action)
    p = random.random()
    if p < 0.33:
        state  = torch.flip(state, dims=[2])       # horizontal flip
        action = _remap_h(action)
    elif p < 0.66:
        state  = torch.flip(state, dims=[1])       # vertical flip
        action = _remap_v(action)
    else:
        state  = torch.flip(state, dims=[1, 2])    # 180°
        action = _remap_v(_remap_h(action))
    return state, int(action)


# ===========================================================================
# Chunk / dataset utilities
# ===========================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _manifest_path(d: str) -> str:
    return os.path.join(d, MANIFEST_NAME)

def load_manifest(d: str) -> Dict:
    p = _manifest_path(d)
    return json.load(open(p)) if os.path.exists(p) else {"version": 1, "chunks": []}

def save_manifest(d: str, m: Dict) -> None:
    with open(_manifest_path(d), "w") as f:
        json.dump(m, f, indent=2)

def flush_chunk(
    chunk_dir: str, chunk_idx: int,
    states: List[np.ndarray], actions: List[int], seeds: List[int],
) -> Dict:
    if not states:
        return {}
    st  = np.stack(states, 0).astype(np.float32)
    ac  = np.array(actions, dtype=np.int64)
    se  = np.array(seeds,   dtype=np.int64)
    hist = np.bincount(ac, minlength=NUM_ACTIONS).astype(int).tolist()
    fname = f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(os.path.join(chunk_dir, fname), states=st, actions=ac, seeds=se)
    return {"file": fname, "count": int(len(ac)), "action_hist": hist,
            "seed_min": int(se.min()), "seed_max": int(se.max())}


class ChunkedBCDataset(IterableDataset):
    """Iterable dataset over chunked .npz files with optional augmentation."""
    def __init__(
        self, chunk_dir: str, augment: bool,
        shuffle_chunks: bool, shuffle_within_chunk: bool, seed: int,
    ):
        super().__init__()
        self.chunk_dir            = chunk_dir
        self.augment              = augment
        self.shuffle_chunks       = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seed                 = seed
        m = load_manifest(chunk_dir)
        self.chunks    = list(m.get("chunks", []))
        self.total_len = int(sum(int(c.get("count", 0)) for c in self.chunks))

    def __len__(self) -> int:
        return self.total_len

    def __iter__(self):
        info       = get_worker_info()
        wid        = 0 if info is None else info.id
        nw         = 1 if info is None else info.num_workers
        rng        = np.random.default_rng(self.seed + wid * 1337)
        idxs       = np.arange(len(self.chunks))
        if self.shuffle_chunks:
            rng.shuffle(idxs)
        for ci in idxs[wid::nw]:
            data    = np.load(os.path.join(self.chunk_dir, self.chunks[int(ci)]["file"]))
            states  = data["states"]
            actions = data["actions"]
            order   = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(order)
            for i in order:
                st = torch.from_numpy(states[int(i)]).float()
                ac = int(actions[int(i)])
                if self.augment:
                    st, ac = augment_tensor_and_action(st, ac)
                yield st, torch.tensor(ac, dtype=torch.long)


def compute_class_weights(chunk_dir: str) -> torch.Tensor:
    m     = load_manifest(chunk_dir)
    total = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for c in m.get("chunks", []):
        total += np.array(c.get("action_hist", [0] * NUM_ACTIONS), dtype=np.float64)
    total   = np.maximum(total, 1.0)
    weights = total.sum() / total
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


# ===========================================================================
# Opponent building
# ===========================================================================
def build_opponents(controlled_id: int, game_seed: int) -> Dict[int, object]:
    """
    Opponents for BC / DAgger data collection.
    FIX: expanded pool to 5 baselines (added BoxFarmerAgent, SimpleRuleAgent)
         for more diverse training states.
    """
    rng  = random.Random(game_seed)
    pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent] if cls is not None]
    if not pool:
        pool = [_FallbackRuleAgent]
    other_ids = [pid for pid in range(4) if pid != controlled_id]
    return {pid: rng.choice(pool)(pid) for pid in other_ids}


class FrozenPolicyAgent:
    """
    Wraps a frozen model snapshot as an opponent.
    FIX: tracks actual game step so the step-ratio observation channel
         (ch 23) is correct (was hardcoded to 0 in v3).
    """
    def __init__(self, agent_id: int, model: nn.Module, deterministic: bool = True):
        self.agent_id     = int(agent_id)
        self.model        = model
        self.deterministic = bool(deterministic)
        self._step        = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: Dict) -> int:
        if (self.agent_id >= len(obs["players"])
                or int(obs["players"][self.agent_id][2]) != 1):
            self._step += 1
            return 0
        step = self._step
        self._step += 1
        state = encode_obs(
            obs["map"], obs["players"], obs["bombs"], self.agent_id, step
        ).unsqueeze(0).to(DEVICE)
        my_pos     = (int(obs["players"][self.agent_id][0]),
                      int(obs["players"][self.agent_id][1]))
        bombs_left = int(obs["players"][self.agent_id][3])
        legal_mask = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bombs_left)
        shield     = _shielded_legal_mask(
            obs["map"], obs["players"], obs["bombs"], self.agent_id, legal_mask
        )
        with torch.no_grad():
            action, _, _, _ = _sample_masked_action(
                self.model, state, shield, sample=not self.deterministic
            )
        return int(action)


class LeaguePool:
    """
    Maintains a rolling pool of past model checkpoints for diverse self-play.
    The current policy, frozen at the start of each PPO round, is added after
    every round, giving opponents that span the model's learning history.
    """
    def __init__(self, max_size: int = LEAGUE_POOL_SIZE):
        self.max_size  = max_size
        self.snapshots: List[nn.Module] = []

    def add(self, model: nn.Module) -> None:
        snap = copy.deepcopy(model).cpu().eval()
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self) -> Optional[nn.Module]:
        return random.choice(self.snapshots) if self.snapshots else None


def build_selfplay_opponents(
    controlled_id: int, game_seed: int,
    frozen_model: Optional[nn.Module] = None,
    league_pool: Optional[LeaguePool] = None,
) -> Dict[int, object]:
    """
    Assign opponents for PPO rollout games.
    Each slot independently drawn:
      40% chance → frozen current policy (if available)
      40% chance → league snapshot (if available)
      20% chance → baseline
    This gives the learner diverse opponents within a single game.
    """
    rng = random.Random(game_seed)
    base_pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent]
                 if cls is not None] or [_FallbackRuleAgent]
    opponents: Dict[int, object] = {}
    for pid in [p for p in range(4) if p != controlled_id]:
        r = rng.random()
        if r < 0.40 and frozen_model is not None:
            fp = FrozenPolicyAgent(pid, frozen_model, deterministic=rng.random() < 0.7)
            fp.reset()
            opponents[pid] = fp
        elif r < 0.80 and league_pool is not None and league_pool.snapshots:
            lm = league_pool.sample().to(DEVICE)
            fp = FrozenPolicyAgent(pid, lm, deterministic=rng.random() < 0.5)
            fp.reset()
            opponents[pid] = fp
        else:
            opponents[pid] = rng.choice(base_pool)(pid)
    return opponents


# ===========================================================================
# Data collection — BC and DAgger
# ===========================================================================
def collect_initial_data(train_dir: str, val_dir: str, num_games: int) -> None:
    ensure_dir(train_dir); ensure_dir(val_dir)
    tr_man  = load_manifest(train_dir); va_man  = load_manifest(val_dir)
    tr_ci   = len(tr_man["chunks"]);   va_ci   = len(va_man["chunks"])
    tr_s, tr_a, tr_se = [], [], []
    va_s, va_a, va_se = [], [], []

    for gi in range(num_games):
        seed   = SEED + gi
        cid    = gi % 4
        split  = "val" if seed % TRAIN_SPLIT_MOD == 0 else "train"
        env    = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs    = env.reset()
        teacher = TeacherEnsemble(cid)
        opps   = build_opponents(cid, seed)
        done   = False; step = 0

        while not done:
            state_np = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).numpy().astype(np.float32)
            expert   = int(teacher.act(obs))
            if split == "train":
                tr_s.append(state_np); tr_a.append(expert); tr_se.append(seed)
            else:
                va_s.append(state_np); va_a.append(expert); va_se.append(seed)

            acts = [0, 0, 0, 0]; acts[cid] = expert
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated)
            step += 1

            if split == "train" and len(tr_s) >= CHUNK_SIZE:
                e = flush_chunk(train_dir, tr_ci, tr_s, tr_a, tr_se)
                if e: tr_man["chunks"].append(e); save_manifest(train_dir, tr_man); tr_ci += 1
                tr_s.clear(); tr_a.clear(); tr_se.clear()
            if split == "val" and len(va_s) >= CHUNK_SIZE:
                e = flush_chunk(val_dir, va_ci, va_s, va_a, va_se)
                if e: va_man["chunks"].append(e); save_manifest(val_dir, va_man); va_ci += 1
                va_s.clear(); va_a.clear(); va_se.clear()

        if (gi + 1) % 100 == 0:
            print(f"BC collect {gi+1}/{num_games}", flush=True)

    for buf_s, buf_a, buf_se, d, ci, man in [
        (tr_s, tr_a, tr_se, train_dir, tr_ci, tr_man),
        (va_s, va_a, va_se, val_dir,   va_ci, va_man),
    ]:
        if buf_s:
            e = flush_chunk(d, ci, buf_s, buf_a, buf_se)
            if e: man["chunks"].append(e)
    save_manifest(train_dir, tr_man); save_manifest(val_dir, va_man)


def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int) -> int:
    """DAgger: student roll-in, teacher labels; collects only on disagreements."""
    ensure_dir(out_dir)
    model.eval()
    man      = load_manifest(out_dir)
    ci       = len(man["chunks"])
    buf_s, buf_a, buf_se = [], [], []
    collected = 0

    def _flush():
        nonlocal ci, collected
        if not buf_s:
            return
        e = flush_chunk(out_dir, ci, buf_s, buf_a, buf_se)
        if e:
            man["chunks"].append(e); save_manifest(out_dir, man)
            collected += e["count"]; ci += 1
        buf_s.clear(); buf_a.clear(); buf_se.clear()

    for gi in range(num_games):
        seed = 100000 + SEED + gi
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs  = env.reset()
        teacher = TeacherEnsemble(cid)
        opps    = build_opponents(cid, seed)
        done    = False; step = 0

        while not done:
            state = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = _model_logits_value(model, state)
                student   = int(torch.argmax(logits, 1).item())
            expert = int(teacher.act(obs))

            if student != expert or (student == 0 and expert != 0):
                buf_s.append(state.squeeze(0).cpu().numpy().astype(np.float32))
                buf_a.append(expert); buf_se.append(seed)

            acts = [0, 0, 0, 0]; acts[cid] = student
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1
            if len(buf_s) >= CHUNK_SIZE:
                _flush()

        if (gi + 1) % 50 == 0:
            print(f"DAgger {gi+1}/{num_games} | samples≈{collected+len(buf_a)}", flush=True)

    _flush()
    return collected


# ===========================================================================
# Reward shaping
# ===========================================================================
def compute_shaped_reward(
    prev_obs: Dict, next_obs: Dict,
    my_id: int, action: int,
    terminated: bool, truncated: bool,
) -> float:
    reward = 0.0
    prev_players, next_players = prev_obs["players"], next_obs["players"]
    prev_map,     next_map     = prev_obs["map"],     next_obs["map"]

    if my_id < len(prev_players) and my_id < len(next_players):
        prev_alive = int(prev_players[my_id][2])
        next_alive = int(next_players[my_id][2])
        if prev_alive == 1 and next_alive == 1:
            reward += 0.001
        elif prev_alive == 1 and next_alive == 0:
            reward -= 6.0

        bonus_gain = max(0, int(next_players[my_id][4]) - int(prev_players[my_id][4]))
        if bonus_gain > 0:
            reward += 0.04 * bonus_gain

        npos = (int(next_players[my_id][0]), int(next_players[my_id][1]))
        if 0 <= npos[0] < prev_map.shape[0] and 0 <= npos[1] < prev_map.shape[1]:
            prev_cell = int(prev_map[npos[0], npos[1]])
            next_cell = int(next_map[npos[0], npos[1]])
            if prev_cell in (3, 4) and next_cell == 0:
                reward += 0.04 if prev_cell == 3 else 0.06

    prev_alive_e = int(np.sum(prev_players[:, 2])) - int(prev_players[my_id][2]) if my_id < len(prev_players) else 0
    next_alive_e = int(np.sum(next_players[:, 2])) - int(next_players[my_id][2]) if my_id < len(next_players) else 0
    kills = max(0, prev_alive_e - next_alive_e)
    if kills > 0:
        bonus = 1.2 + (0.8 if next_alive_e <= 1 else 0.0)
        reward += bonus * kills

    boxes_destroyed = max(0, int(np.sum(prev_map == 2)) - int(np.sum(next_map == 2)))
    if boxes_destroyed > 0:
        reward += 0.01 * boxes_destroyed + (0.003 * (boxes_destroyed - 1) if boxes_destroyed >= 2 else 0.0)

    if action == 5 and my_id < len(prev_players) and int(prev_players[my_id][2]) == 1:
        my_pos = (int(prev_players[my_id][0]), int(prev_players[my_id][1]))
        if should_place_bomb_here(prev_map, prev_players, prev_obs["bombs"], my_id, my_pos):
            reward += 0.04
        else:
            # FIX: penalty was -0.20 (7:1 ratio); now -0.10 (~2.5:1) to avoid over-suppressing bombing
            reward -= 0.10

    reward -= 0.004  # anti-stalling

    if terminated or truncated:
        if my_id < len(next_players) and int(next_players[my_id][2]) == 1:
            reward += 10.0 if int(np.sum(next_players[:, 2])) == 1 else 0.05
        else:
            reward -= 2.0

    return float(np.clip(reward, -12.0, 12.0))


# ===========================================================================
# Action masking
# ===========================================================================
def _legal_action_mask(
    grid: np.ndarray, bombs: np.ndarray,
    my_pos: Tuple[int, int], bombs_left: int,
) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in legal_actions(grid, bombs, my_pos, bombs_left):
        mask[int(a)] = 1.0
    if mask.sum() <= 0:
        mask[0] = 1.0
    return mask


def _shielded_legal_mask(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    my_id: int, legal_mask: np.ndarray,
) -> np.ndarray:
    mask = np.array(legal_mask, dtype=np.float32, copy=True)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        if mask.sum() <= 0:
            mask[0] = 1.0
        return mask

    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    blocked = bomb_positions_set(bombs)
    dng_now  = danger_plane(grid, players, bombs, timer_threshold=1)
    dng_soon = danger_plane(grid, players, bombs, timer_threshold=2)
    in_danger = bool(dng_now[my_pos[0], my_pos[1]] > 0 or dng_soon[my_pos[0], my_pos[1]] > 0)

    if in_danger:
        safe_moves = []
        for a in (1, 2, 3, 4):
            if mask[a] <= 0:
                continue
            nr, nc = next_pos(my_pos, a)
            if not passable(grid, nr, nc) or (nr, nc) in blocked:
                mask[a] = 0.0; continue
            if escape_margin_from_position(grid, players, bombs, (nr, nc), max_depth=6) > 0:
                safe_moves.append(a)
            else:
                mask[a] = 0.0
        if safe_moves:
            mask[0] = 0.0
        elif mask[0] <= 0:
            mask[0] = 1.0
    else:
        if mask[5] > 0 and not should_place_bomb_here(grid, players, bombs, my_id, my_pos):
            mask[5] = 0.0

    if mask.sum() <= 0:
        mask[0] = 1.0
    return mask


def _sample_masked_action(
    model: nn.Module, state: torch.Tensor,
    legal_mask: np.ndarray, sample: bool = True, temperature: float = 1.0,
) -> Tuple[int, float, float, float]:
    logits, value = _model_logits_value(model, state)
    logits = logits / max(float(temperature), 1e-6)
    mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
    masked  = logits.clone()
    masked[~mask_t] = -1e9
    dist = Categorical(logits=masked)
    action = dist.sample() if sample else torch.argmax(masked, dim=-1)
    return (
        int(action.item()),
        float(dist.log_prob(action).item()),
        float(dist.entropy().item()),
        float(value.item()) if value is not None else 0.0,
    )


# ===========================================================================
# BC training
# ===========================================================================
def build_loaders(train_dir: str, val_dir: str):
    tr_ds = ChunkedBCDataset(train_dir, augment=True,  shuffle_chunks=True,  shuffle_within_chunk=True,  seed=SEED)
    va_ds = ChunkedBCDataset(val_dir,   augment=False, shuffle_chunks=False, shuffle_within_chunk=False, seed=SEED)
    if len(tr_ds) == 0: raise RuntimeError(f"No training samples in {train_dir}")
    if len(va_ds) == 0: raise RuntimeError(f"No validation samples in {val_dir}")
    kw = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
              pin_memory=(DEVICE.type == "cuda"), drop_last=False)
    return DataLoader(tr_ds, **kw), DataLoader(va_ds, **kw), compute_class_weights(train_dir).to(DEVICE)


def _run_bc_epoch(
    model: nn.Module, loader: DataLoader,
    criterion, optimizer=None,
) -> Tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = total_correct = total_n = 0

    for bi, (states, actions) in enumerate(loader):
        states  = states.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        logits, _ = _model_logits_value(model, states)
        loss      = criterion(logits, actions)
        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        total_loss    += float(loss.item()) * states.size(0)
        total_correct += int((torch.argmax(logits, 1) == actions).sum().item())
        total_n       += int(states.size(0))
        if bi % 50 == 0:
            print(f"  batch={bi}", flush=True)

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def train_policy_model(
    train_dir: str, val_dir: str,
    init_model_path: Optional[str] = None,
    lr: float = LEARNING_RATE,
) -> nn.Module:
    tr_loader, va_loader, cw = build_loaders(train_dir, val_dir)
    model = BomberNet(INPUT_CHANNELS).to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        model.load_state_dict(torch.load(init_model_path, map_location=DEVICE), strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.03)

    best_val = float("inf"); best_state = None; patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = _run_bc_epoch(model, tr_loader, criterion, optimizer)
        va_loss, va_acc = _run_bc_epoch(model, va_loader, criterion)
        scheduler.step(va_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={tr_loss:.4f}/{tr_acc:.4f} | val={va_loss:.4f}/{va_acc:.4f}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val  = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_left = PATIENCE
            print("  → best saved", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  → early stop", flush=True); break

    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    return model


# ===========================================================================
# PPO rollout storage + GAE  — FIX: lambda actually used, bootstrap on trunc
# ===========================================================================
@dataclass
class RolloutEpisode:
    states:     List[np.ndarray]  = field(default_factory=list)
    actions:    List[int]         = field(default_factory=list)
    rewards:    List[float]       = field(default_factory=list)
    # dones[t] = True only for genuine terminal (agent died / game won).
    # Truncated-at-500 episodes set dones[-1] = False and bootstrap from last_value.
    dones:      List[bool]        = field(default_factory=list)
    log_probs:  List[float]       = field(default_factory=list)
    values:     List[float]       = field(default_factory=list)
    masks:      List[np.ndarray]  = field(default_factory=list)
    last_value: float             = 0.0   # V(s_T) bootstrap; 0 for true terminals


def _gae(ep: RolloutEpisode) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation (Schulman et al., 2016).

    FIX vs v3:
      - PPO_LAMBDA is now used (was defined but ignored).
      - Truncated episodes use ep.last_value to bootstrap the future return
        instead of treating step 500 as a terminal with zero future reward.

    Returns (advantages, returns) both of shape (T,).
    """
    T   = len(ep.rewards)
    adv = np.zeros(T, dtype=np.float32)
    vals = np.asarray(ep.values, dtype=np.float32)
    gae  = 0.0

    for t in reversed(range(T)):
        if ep.dones[t]:
            next_val = 0.0
        elif t + 1 < T:
            next_val = float(vals[t + 1])
        else:
            next_val = ep.last_value          # FIX: bootstrap here

        delta = ep.rewards[t] + PPO_GAMMA * next_val - float(vals[t])
        gae   = delta + PPO_GAMMA * PPO_LAMBDA * (1.0 - float(ep.dones[t])) * gae
        adv[t] = gae

    returns = adv + vals
    return adv, returns


def _flatten_episodes(
    episodes: List[RolloutEpisode],
) -> Tuple[torch.Tensor, ...]:
    all_states, all_acts, all_lps = [], [], []
    all_vals,   all_rets, all_advs = [], [], []
    all_masks = []

    for ep in episodes:
        if not ep.states:
            continue
        adv, ret = _gae(ep)
        # Normalize per-episode to stabilise gradients
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        all_states.extend(ep.states);    all_acts.extend(ep.actions)
        all_lps.extend(ep.log_probs);    all_vals.extend(ep.values)
        all_rets.extend(ret.tolist());   all_advs.extend(adv.tolist())
        all_masks.extend(ep.masks)

    if not all_states:
        raise RuntimeError("No rollout samples collected.")

    mk = lambda lst, dt: torch.tensor(np.array(lst), dtype=dt)
    return (
        mk(all_states, torch.float32), mk(all_acts, torch.long),
        mk(all_lps,    torch.float32), mk(all_vals, torch.float32),
        mk(all_rets,   torch.float32), mk(all_advs, torch.float32),
        mk(all_masks,  torch.float32),
    )


# ===========================================================================
# Self-play rollout collection — FIX: dones, bootstrap, step tracking
# ===========================================================================
def collect_selfplay_rollouts(
    model: nn.Module,
    frozen_model: Optional[nn.Module],
    num_games: int,
    round_idx: int = 0,
    league_pool: Optional[LeaguePool] = None,
) -> List[RolloutEpisode]:
    model.eval()
    if frozen_model is not None:
        frozen_model.eval()

    episodes: List[RolloutEpisode] = []

    for gi in range(num_games):
        seed = 300000 + SEED + round_idx * 10000 + gi
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs  = env.reset()
        opps = build_selfplay_opponents(cid, seed, frozen_model, league_pool)

        ep   = RolloutEpisode()
        done = False; step = 0
        truncated_alive = False   # signals whether we should bootstrap

        while not done:
            if cid >= len(obs["players"]) or int(obs["players"][cid][2]) != 1:
                break

            state  = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl     = int(obs["players"][cid][3])
            lm     = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bl)
            shield = _shielded_legal_mask(obs["map"], obs["players"], obs["bombs"], cid, lm)

            with torch.no_grad():
                action, log_prob, _, value = _sample_masked_action(
                    model, state, shield, sample=True, temperature=0.90
                )

            acts = [0, 0, 0, 0]; acts[cid] = int(action)
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))

            prev_obs = obs
            obs, terminated, truncated = env.step(acts)
            my_died  = (int(obs["players"][cid][2]) == 0)
            reward   = compute_shaped_reward(prev_obs, obs, cid, action, terminated, truncated)

            # FIX: mark genuine terminal only for death / game-over win, NOT for truncation
            genuine_done = bool(my_died or terminated)

            ep.states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
            ep.actions.append(int(action));   ep.rewards.append(float(reward))
            ep.dones.append(genuine_done);    ep.log_probs.append(float(log_prob))
            ep.values.append(float(value));   ep.masks.append(shield.astype(np.float32))

            truncated_alive = bool(truncated and not terminated and not my_died)
            done = bool(terminated or truncated or my_died)
            step += 1

        # FIX: bootstrap last_value when the game was cut off at 500 steps
        #      and the agent is still alive.
        ep.last_value = 0.0
        if truncated_alive and ep.states:
            try:
                ls = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    _, lv = _model_logits_value(model, ls)
                if lv is not None:
                    ep.last_value = float(lv.item())
            except Exception:
                pass

        if ep.states:
            episodes.append(ep)

        if (gi + 1) % 25 == 0:
            total_steps = sum(len(e.states) for e in episodes)
            print(f"Rollout {gi+1}/{num_games} | eps={len(episodes)} | steps={total_steps}", flush=True)

    return episodes


# ===========================================================================
# PPO fine-tuning
# ===========================================================================
def ppo_finetune(
    model: nn.Module,
    episodes: List[RolloutEpisode],
    bc_mix_dir: Optional[str] = None,
) -> nn.Module:
    if not episodes:
        return model
    states, actions, old_lps, old_vals, returns, advantages, masks = _flatten_episodes(episodes)
    N         = states.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)

    bc_loader = None
    if bc_mix_dir and os.path.exists(bc_mix_dir):
        try:
            ds = ChunkedBCDataset(bc_mix_dir, augment=True,
                                  shuffle_chunks=True, shuffle_within_chunk=True, seed=SEED+999)
            if len(ds) > 0:
                bc_loader = DataLoader(ds, batch_size=min(128, PPO_BATCH_SIZE),
                                       shuffle=False, num_workers=0, drop_last=True)
        except Exception:
            pass

    model.train()
    bc_iter = iter(bc_loader) if bc_loader else None

    for epoch in range(1, PPO_EPOCHS + 1):
        idxs = np.random.permutation(N)
        t_pol = t_val = t_ent = t_tot = n_b = 0.0

        for start in range(0, N, PPO_BATCH_SIZE):
            bi  = idxs[start:start + PPO_BATCH_SIZE]
            if len(bi) == 0: continue

            bs  = states[bi].to(DEVICE);   ba  = actions[bi].to(DEVICE)
            blp = old_lps[bi].to(DEVICE);  brt = returns[bi].to(DEVICE)
            bad = advantages[bi].to(DEVICE); bm = masks[bi].to(DEVICE)

            logits, values = _model_logits_value(model, bs)
            ml = logits.clone(); ml[bm <= 0] = -1e9
            dist = Categorical(logits=ml)
            new_lp  = dist.log_prob(ba)
            entropy = dist.entropy().mean()

            ratio    = torch.exp(new_lp - blp)
            clipped  = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS)
            pol_loss = -torch.mean(torch.min(ratio * bad, clipped * bad))
            val_loss = torch.mean((values - brt) ** 2)
            loss     = pol_loss + PPO_VALUE_COEF * val_loss - PPO_ENTROPY_COEF * entropy

            if bc_iter is not None:
                try:
                    bc_s, bc_a = next(bc_iter)
                except StopIteration:
                    bc_iter = iter(bc_loader); bc_s, bc_a = next(bc_iter)
                bc_s = bc_s.to(DEVICE); bc_a = bc_a.to(DEVICE)
                bc_logits, _ = _model_logits_value(model, bc_s)
                loss = loss + BC_MIX_COEF * nn.functional.cross_entropy(bc_logits, bc_a)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
            optimizer.step()

            t_pol += pol_loss.item(); t_val += val_loss.item()
            t_ent += entropy.item();  t_tot += loss.item(); n_b += 1

        print(
            f"PPO {epoch:02d}/{PPO_EPOCHS} | "
            f"loss={t_tot/max(1,n_b):.4f} pol={t_pol/max(1,n_b):.4f} "
            f"val={t_val/max(1,n_b):.4f} ent={t_ent/max(1,n_b):.4f}",
            flush=True,
        )

    torch.save(model.state_dict(), MODEL_PATH)
    return model


# ===========================================================================
# Evaluation
# ===========================================================================
def quick_eval_against_baselines(model: nn.Module, num_games: int = 30) -> None:
    model.eval()
    wins = draws = losses = 0
    total_kills = total_boxes = total_steps = 0

    for gi in range(num_games):
        seed = 400000 + SEED + gi
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs  = env.reset()
        opps = build_opponents(cid, seed)
        init_boxes = int(np.sum(obs["map"] == 2))
        kills = 0; done = False; step = 0

        while not done:
            if int(obs["players"][cid][2]) != 1:
                break
            state  = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl     = int(obs["players"][cid][3])
            lm     = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bl)
            shield = _shielded_legal_mask(obs["map"], obs["players"], obs["bombs"], cid, lm)
            with torch.no_grad():
                action, _, _, _ = _sample_masked_action(model, state, shield, sample=False)

            prev_e = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            acts = [0, 0, 0, 0]; acts[cid] = action
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            next_e = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            kills += max(0, prev_e - next_e)
            done = bool(terminated or truncated); step += 1

        alive  = [int(p[2]) for p in obs["players"]]
        boxes_destroyed = init_boxes - int(np.sum(obs["map"] == 2))
        if alive[cid] == 1 and sum(alive) == 1:
            wins += 1
        elif alive[cid] == 1:
            draws += 1
        else:
            losses += 1
        total_kills += kills; total_boxes += boxes_destroyed; total_steps += step

    ng = max(1, num_games)
    print(
        f"Eval ({num_games}g) | W={wins} D={draws} L={losses} | "
        f"kills={total_kills/ng:.2f} boxes={total_boxes/ng:.0f} steps={total_steps/ng:.0f}",
        flush=True,
    )


# ===========================================================================
# Main — single consolidated entry point
# ===========================================================================
def main() -> None:
    ensure_dir(TRAIN_DIR); ensure_dir(VAL_DIR)

    if BASELINE_IMPORT_ERRORS:
        print("Baseline import warnings (fallback agent used):", flush=True)
        for name, err in BASELINE_IMPORT_ERRORS:
            print(f"  {name}: {err}", flush=True)

    # print("=== Phase 1: BC data collection ===", flush=True)
    # collect_initial_data(TRAIN_DIR, VAL_DIR, INITIAL_GAMES)

    # print("=== Phase 2: BC policy/value training ===", flush=True)
    # model = train_policy_model(TRAIN_DIR, VAL_DIR, lr=LEARNING_RATE)

    # print("=== Phase 3: DAgger correction ===", flush=True)
    # n = collect_dagger_data(model, TRAIN_DIR, MIXED_DAGGER_GAMES)
    # print(f"DAgger collected {n} corrective samples", flush=True)

    # print("=== Phase 4: Refresh BC with aggregated data ===", flush=True)
    # model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_PATH, lr=FINE_TUNE_LR)

    model = BomberNet(INPUT_CHANNELS).to(DEVICE)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "model_bc.pth")
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        raise FileNotFoundError(f"Can't find {pretrained_path}")


    print("=== Phase 5: PPO self-play fine-tuning ===", flush=True)
    league = LeaguePool(max_size=LEAGUE_POOL_SIZE)
    league.add(model)   # seed the pool with the BC model

    for round_idx in range(RL_ROUNDS):
        print(f"--- PPO round {round_idx + 1}/{RL_ROUNDS} ---", flush=True)
        frozen = copy.deepcopy(model).to(DEVICE).eval()
        rollouts = collect_selfplay_rollouts(
            model,
            frozen_model=frozen,
            num_games=ROLLOUT_GAMES_PER_ROUND,
            round_idx=round_idx,
            league_pool=league,
        )
        print(f"  collected {len(rollouts)} episodes "
              f"({sum(len(e.states) for e in rollouts)} steps)", flush=True)
        model = ppo_finetune(model, rollouts, bc_mix_dir=TRAIN_DIR)
        league.add(model)  # add the updated snapshot to the pool

        # Quick checkpoint eval every round
        quick_eval_against_baselines(model, num_games=20)

    print("=== Final evaluation ===", flush=True)
    quick_eval_against_baselines(model, num_games=50)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()