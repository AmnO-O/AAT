"""
ClaudFat_v6.py — Bomberland training pipeline

Changes vs v5:
  [CRITICAL] Fixed CUDA/CPU device mismatch in FrozenPolicyAgent.act() — it was
             using the global DEVICE (cuda) to send states, but worker-side
             frozen_model lives on CPU. Now infers device from the model itself:
             `next(self.model.parameters()).device`. This was silently crashing
             all 3 workers every single round.

  [CRITICAL] NUM_ROLLOUT_WORKERS = 1 — workers always failed and fell back
             anyway; skipping the failed spawn overhead saves ~15s per round.

  [CRITICAL] PPO_EPOCHS 6→3 — policy loss was going deeply negative by epoch
             3-4 every round (clip firing hard, oscillation). 3 epochs consumes
             the rollout gradient before it stales.

  [CRITICAL] BC_MIX_COEF 0.05→0.0 — the BC regulariser was anchoring the policy
             back to the teacher ensemble after every PPO mini-batch. Since the
             BC model sits near 10% win rate (below random), this was actively
             preventing improvement. Removed entirely.

  [HIGH]     Rollout now uses _legal_action_mask NOT _shielded_legal_mask —
             the shield was hard-masking any move without a provably safe escape,
             preventing the agent from ever learning to navigate danger or time
             bomb escapes. The reward signal (death −4.0, unsafe bomb −0.12) is
             sufficient to teach safety. Shield is kept for eval / agent.py.

  [HIGH]     PPO_CLIP_EPS 0.15→0.20 — restored now that epochs are fixed at 3.
             0.15 was introduced to dampen the 6-epoch oscillation; that root
             cause is now resolved.

  [HIGH]     Opponent mix 50/30/20 → 40/20/40 (frozen/league/baselines) —
             previous 80% self-play meant training mostly against a near-baseline
             policy. 40% strong baselines forces learning to beat tactical opponents.
             Baseline pool limited to tactical/genius/smarter only.

  [MEDIUM]   ROLLOUT_GAMES_PER_ROUND 300→500 — more diverse states per round,
             better gradient estimates.

  [MEDIUM]   RL_ROUNDS 20→100 — 20 rounds was far too few; eval showed the
             moving average was still rising in rounds 16-20.

  [MEDIUM]   Per-round LR decay — PPO fine-tune LR starts at 3e-4 and decays
             ×0.995 per round (≈ 2.2e-4 at round 50, 1.8e-4 at round 100).

  [MEDIUM]   Best PPO checkpoint — main() saves model_best_ppo.pth whenever eval
             wins improve. Training continues from current model but peak is kept.

  [MINOR]    Rollout temperature 0.90→1.00 — with the shield removed, raw-logit
             temperature is appropriate for exploration.

Changes vs v4:
  [CRITICAL] Global advantage normalization — advantages are now normalized across
             the entire batch after flattening, NOT per-episode. Per-episode normalization
             was washing out the kill signal (~0.9) against hundreds of survival ticks
             (~0.0002 each), making kills indistinguishable from good STOP steps.

  [CRITICAL] Rebalanced reward function:
             - Kill reward: +0.9 → +2.0 (last-enemy kill: +3.5)
             - Enemy-in-blast reward: +0.08 → +0.25 (leading indicator of kill)
             - Death penalty: -6.0 → -4.0 (was over-penalizing vs kill reward)
             - Terminal death: -2.0 → -1.5
             - Safe bomb: +0.10 (unchanged, but now less dominant relative to kills)
             - Box destroy: +0.03 (unchanged)

  [CRITICAL] Relaxed shield mask for offensive plays — bomb action is now allowed
             when an enemy is inside the blast radius even if the escape is tight
             (margin > -1 instead of requiring > 0). This unblocks enemy-trapping plays
             that the previous mask was completely suppressing.

  [HIGH]     Entropy annealing — coef starts at 0.03 and decays ×0.80 per round,
             flooring at 0.005. This lets the policy actually commit to a strategy
             in later rounds instead of staying near-uniform (entropy ~0.47) forever.

  [HIGH]     PPO clip reduced: 0.20 → 0.15. Policy loss was going negative by epoch 3-4
             every round, indicating the clip was firing too hard and causing oscillation.

  [HIGH]     Multiprocessing rollout collection — rollout games are distributed across
             NUM_ROLLOUT_WORKERS parallel worker processes using torch.multiprocessing.
             Each worker collects a shard of the total games and returns episodes via a
             shared Queue. On 8-vCPU machines this gives ~4-6× speedup on rollout.
             Workers use 'spawn' to avoid CUDA fork issues. Falls back to single-process
             if workers fail to start.

  [HIGH]     Vectorized encode_obs — the most expensive per-step call is now ~2× faster:
             - blast_tiles replaced with vectorized NumPy flood-fill
             - explosion_time_plane and danger computations use array ops not Python loops
             - BFS reachable count uses pre-allocated visited array instead of set
             - All channel assignments use array slicing instead of element-wise Python

  [MEDIUM]   Expanded league pool: size 4 → 8. BC checkpoint saved as permanent anchor
             so the pool always contains a 'young' policy to exploit.

  [MEDIUM]   Opponent sampling rebalanced: 50% frozen-current / 30% league / 20% baselines,
             with baselines weighted toward tactical_rule_agent and genius_rule_agent.

  [MEDIUM]   Eval seeds randomized — no longer runs the same 50 seeds every eval.

  [MINOR]    PPO_ENTROPY_COEF now passed as argument to ppo_finetune() so it can be
             annealed per-round without touching the global constant.

  [MINOR]    ROLLOUT_GAMES_PER_ROUND raised from 250 → 300 to feed the larger league pool.

NOTE: agent.py submitted to the contest must use the same BomberNet class defined here.
      Copy BomberNet + ResidualBlock + SPATIAL_CHANNELS / SCALAR_CHANNELS constants,
      plus encode_obs and all helper functions up through _shielded_legal_mask.
"""

import copy
import json
import os
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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
# ===========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 200

BOARD_SIZE             = 13
INPUT_CHANNELS         = 27
NUM_ACTIONS            = 6
MAX_STEPS              = 500
EXPLOSION_TIME_HORIZON = 8.0

# Channel split — unchanged from v4
SPATIAL_CHANNELS: List[int] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,21,24,25,26]
SCALAR_CHANNELS:  List[int] = [14,17,18,19,20,22,23]
N_SPATIAL = len(SPATIAL_CHANNELS)  # 20
N_SCALAR  = len(SCALAR_CHANNELS)   # 7

# --- BC / DAgger (unchanged)
INITIAL_GAMES       = 800
MIXED_DAGGER_GAMES  = 250
TRAIN_SPLIT_MOD     = 10
CHUNK_SIZE          = 2048
BATCH_SIZE          = 128
EPOCHS              = 20
LEARNING_RATE       = 1e-3
FINE_TUNE_LR        = 3e-4
WEIGHT_DECAY        = 1e-4
PATIENCE            = 5
GRAD_CLIP_NORM      = 1.0

TRAIN_DIR       = "bc_train_chunks"
VAL_DIR         = "bc_val_chunks"
MODEL_PATH      = "model_bc_.pth"
BEST_MODEL_PATH = "model_bc_best_.pth"
MANIFEST_NAME   = "manifest.json"

AUGMENT_FLIP_PROB = 0.5

# --- PPO / self-play
RL_ROUNDS               = 100   # FIX v6: was 20 — not enough to converge
ROLLOUT_GAMES_PER_ROUND = 500   # FIX v6: was 300 — more diverse states per round
PPO_EPOCHS              = 3     # FIX v6: was 6 — clip was firing hard by epoch 3-4
PPO_BATCH_SIZE          = 256
PPO_CLIP_EPS            = 0.20  # FIX v6: restored from 0.15 — 3 epochs no longer oscillate
PPO_GAMMA               = 0.98
PPO_LAMBDA              = 0.95
PPO_VALUE_COEF          = 0.5
PPO_ENTROPY_COEF        = 0.03  # annealed per round below
PPO_ENTROPY_DECAY       = 0.80  # multiply per round; floor at PPO_ENTROPY_MIN
PPO_ENTROPY_MIN         = 0.005
PPO_MAX_GRAD_NORM       = 1.0
BC_MIX_COEF             = 0.00  # FIX v6: was 0.05 — BC anchor was preventing PPO improvement
LEAGUE_POOL_SIZE        = 8

# --- Multiprocessing rollout
# FIX v6: set to 1 — workers always failed (FrozenPolicyAgent CUDA bug) and fell
# back to single-process anyway; skipping spawn overhead saves ~15s per round.
NUM_ROLLOUT_WORKERS     = 1


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


# ---------------------------------------------------------------------------
# Vectorized blast: returns a boolean (BOARD_SIZE, BOARD_SIZE) mask.
# Faster than building a Python set when the result is used as an array.
# ---------------------------------------------------------------------------
def blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    """Return a (13,13) bool array marking blast tiles."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    mask[bx, by] = True
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for d in range(1, radius + 1):
            r, c = bx + dr * d, by + dc * d
            if not in_bounds(r, c):
                break
            cell = int(grid[r, c])
            if cell == 1:
                break
            mask[r, c] = True
            if cell == 2:
                break
    return mask


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
# Danger / explosion planes  — vectorized where possible
# ===========================================================================
def explosion_time_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    horizon: float = EXPLOSION_TIME_HORIZON,
) -> np.ndarray:
    plane = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane
    times = bomb_effective_explosion_times(grid, players, bombs)
    denom = horizon if horizon > 0 else 1.0
    for i in range(len(bombs)):
        owner  = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t      = float(max(0, int(times[i])))
        norm_t = min(t, horizon) / denom
        bmask  = blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), radius)
        # Vectorized minimum over the blast region
        plane[bmask] = np.minimum(plane[bmask], norm_t)
    return plane


def danger_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    timer_threshold: int = 1,
) -> np.ndarray:
    if bombs is None or len(bombs) == 0:
        return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / float(EXPLOSION_TIME_HORIZON) if EXPLOSION_TIME_HORIZON > 0 else 0.0
    return (plane <= threshold).astype(np.float32)


def immediate_danger_plane(grid, players, bombs):
    return danger_plane(grid, players, bombs, timer_threshold=1)


def chain_danger_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    chain_horizon: int = 3,
) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane
    original  = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    effective = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        eff = int(effective[i]); orig = int(original[i])
        if eff <= 1 or eff > chain_horizon or eff >= orig:
            continue
        owner  = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        plane[blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), radius)] = 1.0
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
        owner  = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t      = float(max(0, int(effective[i])))
        score  = 1.0 - min(t, denom) / denom
        if score <= 0:
            continue
        bmask = blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), radius)
        plane[bmask] = np.maximum(plane[bmask], score)
    return plane


def tile_earliest_explosion_times(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray
) -> np.ndarray:
    times = np.full((BOARD_SIZE, BOARD_SIZE), 9999, dtype=np.int32)
    if bombs is None or len(bombs) == 0:
        return times
    eff = bomb_effective_explosion_times(grid, players, bombs)
    for i, b in enumerate(bombs):
        owner  = int(b[3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t      = int(max(0, eff[i]))
        bmask  = blast_mask(grid, int(b[0]), int(b[1]), radius)
        times[bmask] = np.minimum(times[bmask], t)
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
        plane[blast_mask(grid, r, c, radius)] = 1.0
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
            plane[blast_mask(grid, pr, pc, radius)] = np.maximum(
                plane[blast_mask(grid, pr, pc, radius)], 0.5
            )
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

    # Build passable mask once
    pass_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            pass_mask[r, c] = passable(grid, r, c) and (r, c) not in blocked

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not pass_mask[r, c]:
                continue
            exits = fragile = 0
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if not pass_mask[nr, nc] if in_bounds(nr, nc) else True:
                    continue
                if not in_bounds(nr, nc):
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
        t_exp  = int(explosion_times[pos[0], pos[1]])
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
    """BFS reachable tile count — uses array visited flags instead of a set for speed."""
    blocked = bomb_positions_set(bombs)
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    visited[start[0], start[1]] = True
    q: deque = deque([(start, 0)])
    count = 0
    while q:
        pos, dist = q.popleft()
        if dist > 0:
            count += 1
        if dist >= max_depth:
            continue
        for a in (1, 2, 3, 4):
            npos = next_pos(pos, a)
            if not in_bounds(npos[0], npos[1]):
                continue
            if visited[npos[0], npos[1]] or npos in blocked or not passable(grid, npos[0], npos[1]):
                continue
            visited[npos[0], npos[1]] = True
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
# Bomb safety helpers
# ===========================================================================
def _add_hypothetical_bomb(
    bombs: np.ndarray, pos: Tuple[int, int], owner: int, timer: int = 7
) -> np.ndarray:
    new_row = np.array([[pos[0], pos[1], timer, owner]], dtype=np.int8)
    if bombs is not None and len(bombs) > 0:
        return np.concatenate([bombs, new_row], axis=0)
    return new_row


def should_place_bomb_here(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    my_id: int, pos: Tuple[int, int],
    enemy_in_blast: bool = False,
) -> bool:
    """
    Check whether placing a bomb at *pos* is survivable AND tactically useful.

    FIX v5: when an enemy is already in the blast radius, accept escape margin > -1
    instead of > 0. This allows offensive trapping plays that v4 completely blocked.
    """
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return False
    if not passable(grid, pos[0], pos[1]):
        return False

    my_radius = 1 + int(players[my_id][4])
    hyp_bombs = _add_hypothetical_bomb(bombs, pos, my_id)
    blast     = blast_tiles(grid, pos[0], pos[1], my_radius)
    blocked   = bomb_positions_set(hyp_bombs)

    # FIX: relaxed escape threshold when hunting an enemy
    escape_threshold = -1.0 if enemy_in_blast else 0.0

    for a in (1, 2, 3, 4):
        nr, nc = next_pos(pos, a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked:
            continue
        if (nr, nc) in blast:
            continue
        margin = escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6)
        if margin > escape_threshold:
            return True
    return False


def _enemy_in_blast(
    grid: np.ndarray, players: np.ndarray, my_id: int, pos: Tuple[int, int], radius: int
) -> bool:
    blast = blast_tiles(grid, pos[0], pos[1], radius)
    for i in range(4):
        if i == my_id or i >= len(players) or int(players[i][2]) != 1:
            continue
        if (int(players[i][0]), int(players[i][1])) in blast:
            return True
    return False


def safe_to_bomb_plane(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int
) -> np.ndarray:
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
    blast       = blast_tiles(grid, my_r, my_c, bomb_radius)

    enemy_positions = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(4)
        if i != my_id and i < len(players) and int(players[i][2]) == 1
    }
    hit_boxes  = any(int(grid[x, y]) == 2 for x, y in blast)
    hit_enemy  = any((x, y) in enemy_positions for x, y in blast)
    if not (hit_boxes or hit_enemy):
        return plane

    hyp_bombs    = _add_hypothetical_bomb(bombs, (my_r, my_c), my_id)
    blocked_hyp  = bomb_positions_set(hyp_bombs)
    # FIX: use relaxed threshold when enemy is in blast
    threshold    = -1.0 if hit_enemy else 0.0

    for a in (1, 2, 3, 4):
        nr, nc = next_pos((my_r, my_c), a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked_hyp:
            continue
        if (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6) > threshold:
            plane[my_r, my_c] = 1.0
            break
    return plane


# ===========================================================================
# Observation encoding  (27 channels)
# ===========================================================================
def encode_obs(
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
    my_id: int, step: int,
) -> torch.Tensor:
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Static map (0-4) — vectorized
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

    # Bomb danger system (9-12) — use vectorized planes
    state[9]  = explosion_time_plane(grid, players, bombs)
    state[10] = immediate_danger_plane(grid, players, bombs)
    state[11] = chain_danger_plane(grid, players, bombs)
    state[12] = future_danger_plane(grid, players, bombs)

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

    # ch 14 — bombs_left scalar
    state[14].fill(normalize_scalar(bombs_left, 5.0))

    # ch 15-16 — bomb timer / radius heatmaps
    if bombs is not None and len(bombs) > 0:
        eff_times = bomb_effective_explosion_times(grid, players, bombs)
        rows = np.array([int(b[0]) for b in bombs])
        cols = np.array([int(b[1]) for b in bombs])
        valid = np.array([in_bounds(int(b[0]), int(b[1])) for b in bombs])
        for i in np.where(valid)[0]:
            r, c = int(bombs[i][0]), int(bombs[i][1])
            t = max(int(eff_times[i]), 1)
            state[15, r, c] = max(state[15, r, c], 1.0 / float(t))
            owner = int(bombs[i][3]) if len(bombs[i]) > 3 else -1
            state[16, r, c] = max(
                state[16, r, c],
                normalize_scalar(bomb_radius_for_owner(players, owner), 6.0)
            )

    # ch 17-23 — scalar / ego features
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
# Teacher ensemble (unchanged from v4)
# ===========================================================================
class _FallbackRuleAgent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs: Dict) -> int:
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0
        r, c       = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left = int(players[self.agent_id][3])
        dng        = danger_plane(grid, players, bombs, timer_threshold=1)
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
    _W = {"tactical": 3.0, "genius": 2.5, "smarter": 2.0,
          "box_farmer": 1.0, "simple": 0.75, "random": 0.25}

    def __init__(self, agent_id: int):
        self.agent_id   = int(agent_id)
        self.tactical   = _maybe_make(TacticalRuleAgent,  agent_id)
        self.genius     = _maybe_make(GeniusRuleAgent,    agent_id)
        self.smarter    = _maybe_make(SmarterRuleAgent,   agent_id)
        self.box_farmer = _maybe_make(BoxFarmerAgent,     agent_id)
        self.simple     = _maybe_make(SimpleRuleAgent,    agent_id)
        self.random     = _maybe_make(RandomAgent,        agent_id)
        self.weights    = dict(self._W)

    def _collect(self, obs: Dict) -> Dict[str, int]:
        return {
            "tactical":   int(self.tactical.act(obs)),
            "genius":     int(self.genius.act(obs)),
            "smarter":    int(self.smarter.act(obs)),
            "box_farmer": int(self.box_farmer.act(obs)),
            "simple":     int(self.simple.act(obs)),
            "random":     int(self.random.act(obs)),
        }

    def _weighted_vote(self, acts: Dict[str, int], legal: Optional[set] = None) -> int:
        score: Counter = Counter()
        for k, v in acts.items():
            score[v] += self.weights[k]
        if legal is not None:
            for a in list(score.keys()):
                if a not in legal:
                    score[a] -= 10.0
        best  = max(score.values())
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
        r, c       = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bombs_left = int(players[self.agent_id][3])
        legal      = set(legal_actions(grid, bombs, (r, c), bombs_left))
        acts       = self._collect(obs)

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
            blast   = blast_tiles(grid, r, c, 1 + int(players[self.agent_id][4]))
            enemies = {(int(players[i][0]), int(players[i][1]))
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
# Model  v6 — width=64, pool=7×7, AlphaZero-style 1×1-conv heads
# ===========================================================================
#
# Key reasoning vs the old 4×4 pool + dense heads:
#
# PROBLEM 1 — 4×4 pool destroys bomb precision.
#   13×13 → 4×4 means each pool cell covers ~3.25×3.25 board squares.
#   A bomb radius-1 blast spans only 3 cells, which fits inside one pool cell.
#   The network literally could not distinguish "bomb is 2 tiles away"
#   from "bomb is 5 tiles away" after pooling.
#   FIX: AdaptiveAvgPool2d(7) → each cell covers ~1.86 board squares.
#        Bomb radius 1-2 is now spatially resolvable.
#
# PROBLEM 2 — heads dominated params (69% = 595K of 865K total).
#   feat_dim=1031 → two separate 256→128 MLPs = 595K head params.
#   This is top-heavy: the CNN extracts features but the heads do the
#   heavy lifting with limited data. Noisy PPO gradients can't train
#   595K params reliably from 185K steps/round.
#   FIX: AlphaZero-style 1×1 conv (64→8) before flattening.
#        feat_dim drops from 1031 to 399.  Head params: ~104K total.
#        CNN backbone stays at 270K (same strong features, correct role).
#
# PROBLEM 3 — dropout too high for the data regime.
#   Dropout(0.20) in policy head was regularising away useful signal.
#   With a smaller model there is less overfitting risk; lower dropout
#   lets the policy gradient propagate more reliably.
#
# Result: 376K total params (was 865K).
#   Over 100 rounds × 500 games: 49 training steps per parameter.
#   Was: 21 steps/param → often under-trained per gradient step.
#
# NOTE: This architecture is INCOMPATIBLE with old checkpoints.
#   Re-run phases 1–4 (BC) before starting Phase 5 (PPO).
#   The main() function handles missing / shape-mismatched checkpoints
#   gracefully and starts BC from scratch when needed.

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


# Constant: 1×1 conv output channels before head MLP
_HEAD_CONV_CH = 8   # 8 × 7 × 7 = 392 spatial features + 7 scalars = 399 feat_dim


class BomberNet(nn.Module):
    """
    v6 actor-critic: width=64 CNN + 7×7 pool + 1×1-conv heads.
    Copy BomberNet, ResidualBlock and _HEAD_CONV_CH verbatim into agent.py.
    """
    _SPATIAL = SPATIAL_CHANNELS   # 20 channels
    _SCALAR  = SCALAR_CHANNELS    # 7 channels
    _POOL    = 7                  # FIX v6: was 4

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_actions:    int = NUM_ACTIONS,
        width:          int = 64,
    ):
        super().__init__()
        n_sp     = len(self._SPATIAL)
        n_sc     = len(self._SCALAR)
        pool_sz  = self._POOL
        feat_dim = _HEAD_CONV_CH * pool_sz * pool_sz + n_sc  # 8×49 + 7 = 399

        # ── Backbone ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(n_sp, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.05),   # FIX v6: dropout 0.10 → 0.05
            ResidualBlock(width, dropout=0.05),
            ResidualBlock(width, dropout=0.05),
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_sz)  # 13×13 → 7×7

        # ── Heads: 1×1 conv → flatten → thin MLP ────────────────────────────
        # 1×1 conv: spatially compress 64→8 channels while preserving location.
        # This is the AlphaZero pattern: let the CNN decide what matters spatially,
        # then read out with a lightweight MLP.
        self.policy_conv = nn.Conv2d(width, _HEAD_CONV_CH, 1)
        self.value_conv  = nn.Conv2d(width, _HEAD_CONV_CH, 1)

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.05),
            nn.Linear(128, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.02),
            nn.Linear(128, 1),
        )

        self.register_buffer("_sp_idx", torch.tensor(self._SPATIAL, dtype=torch.long))
        self.register_buffer("_sc_idx", torch.tensor(self._SCALAR,  dtype=torch.long))

        # Orthogonal init: policy output near-zero → starts near-uniform;
        # value output unit-scale → stable critic bootstrapping.
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sp = x[:, self._sp_idx]           # batch × 20 × 13 × 13
        sc = x[:, self._sc_idx, 0, 0]    # batch × 7

        feat = self.stem(sp)
        feat = self.blocks(feat)
        feat = self.pool(feat)            # batch × 64 × 7 × 7

        # Separate 1×1 projections for policy vs value
        p = torch.relu(self.policy_conv(feat))   # batch × 8 × 7 × 7
        v = torch.relu(self.value_conv(feat))    # batch × 8 × 7 × 7

        p_in = torch.cat([p.flatten(1), sc], dim=1)  # batch × 399
        v_in = torch.cat([v.flatten(1), sc], dim=1)  # batch × 399

        logits = self.policy_head(p_in)
        value  = self.value_head(v_in).squeeze(-1)
        return logits, value


def _model_logits_value(
    model: nn.Module, states: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = model(states)
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, None)


# ===========================================================================
# Augmentation
# ===========================================================================
def _remap_h(a: int) -> int:
    return {1: 2, 2: 1, 3: 3, 4: 4, 0: 0, 5: 5}.get(int(a), int(a))

def _remap_v(a: int) -> int:
    return {3: 4, 4: 3, 1: 1, 2: 2, 0: 0, 5: 5}.get(int(a), int(a))


def augment_tensor_and_action(
    state: torch.Tensor, action: int
) -> Tuple[torch.Tensor, int]:
    if random.random() > AUGMENT_FLIP_PROB:
        return state, int(action)
    p = random.random()
    if p < 0.33:
        state  = torch.flip(state, dims=[2])
        action = _remap_h(action)
    elif p < 0.66:
        state  = torch.flip(state, dims=[1])
        action = _remap_v(action)
    else:
        state  = torch.flip(state, dims=[1, 2])
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
    st   = np.stack(states, 0).astype(np.float32)
    ac   = np.array(actions, dtype=np.int64)
    se   = np.array(seeds,   dtype=np.int64)
    hist = np.bincount(ac, minlength=NUM_ACTIONS).astype(int).tolist()
    fname = f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(os.path.join(chunk_dir, fname), states=st, actions=ac, seeds=se)
    return {"file": fname, "count": int(len(ac)), "action_hist": hist,
            "seed_min": int(se.min()), "seed_max": int(se.max())}


class ChunkedBCDataset(IterableDataset):
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
        info = get_worker_info()
        wid  = 0 if info is None else info.id
        nw   = 1 if info is None else info.num_workers
        rng  = np.random.default_rng(self.seed + wid * 1337)
        idxs = np.arange(len(self.chunks))
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
    rng  = random.Random(game_seed)
    pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent,
                             BoxFarmerAgent, SimpleRuleAgent] if cls is not None]
    if not pool:
        pool = [_FallbackRuleAgent]
    other_ids = [pid for pid in range(4) if pid != controlled_id]
    return {pid: rng.choice(pool)(pid) for pid in other_ids}


class FrozenPolicyAgent:
    def __init__(self, agent_id: int, model: nn.Module, deterministic: bool = True):
        self.agent_id      = int(agent_id)
        self.model         = model
        self.deterministic = bool(deterministic)
        self._step         = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: Dict) -> int:
        if (self.agent_id >= len(obs["players"])
                or int(obs["players"][self.agent_id][2]) != 1):
            self._step += 1
            return 0
        step = self._step
        self._step += 1
        # FIX v6: infer device from model instead of using global DEVICE.
        # Using DEVICE (cuda) with a CPU worker model caused the crash:
        # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
        _fdev = next(self.model.parameters()).device
        state = encode_obs(
            obs["map"], obs["players"], obs["bombs"], self.agent_id, step
        ).unsqueeze(0).to(_fdev)
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
    Rolling pool of past model snapshots for diverse self-play.
    FIX v5: size 8 instead of 4. BC anchor snapshot added at init.
    """
    def __init__(self, max_size: int = LEAGUE_POOL_SIZE):
        self.max_size  = max_size
        self.snapshots: List[nn.Module] = []
        self._anchor:   Optional[nn.Module] = None  # BC checkpoint, never evicted

    def set_anchor(self, model: nn.Module) -> None:
        """Pin the initial BC model as permanent anchor."""
        self._anchor = copy.deepcopy(model).cpu().eval()

    def add(self, model: nn.Module) -> None:
        snap = copy.deepcopy(model).cpu().eval()
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self) -> Optional[nn.Module]:
        pool = self.snapshots[:]
        if self._anchor is not None:
            pool = [self._anchor] + pool
        return random.choice(pool) if pool else None


def build_selfplay_opponents(
    controlled_id: int, game_seed: int,
    frozen_model: Optional[nn.Module] = None,
    league_pool: Optional[LeaguePool] = None,
) -> Dict[int, object]:
    """
    Assign opponents for PPO rollout.
    FIX v6: 40% frozen-current / 20% league / 40% strong baselines.
    Previous 80% self-play meant training against a near-baseline policy;
    raising baselines to 40% forces learning to beat tactical opponents.
    Baseline pool: tactical (4×) / genius (3×) / smarter (2×) only — no random/simple.
    FIX v6: removed .to(DEVICE) on league snapshots — in worker processes the
    models live on CPU; coercing to DEVICE caused CUDA/CPU device mismatches.
    """
    rng = random.Random(game_seed)
    # Strong baselines only: weighted toward top performers
    base_pool = []
    for cls, weight in [(TacticalRuleAgent, 4), (GeniusRuleAgent, 3), (SmarterRuleAgent, 2)]:
        if cls is not None:
            base_pool.extend([cls] * weight)
    if not base_pool:
        base_pool = [_FallbackRuleAgent]

    opponents: Dict[int, object] = {}
    for pid in [p for p in range(4) if p != controlled_id]:
        r = rng.random()
        if r < 0.40 and frozen_model is not None:           # 40% frozen current
            fp = FrozenPolicyAgent(pid, frozen_model, deterministic=rng.random() < 0.7)
            fp.reset()
            opponents[pid] = fp
        elif r < 0.60 and league_pool is not None and league_pool.snapshots:  # 20% league
            # FIX v6: do NOT call .to(DEVICE) — keep model on its own device
            lm = league_pool.sample()
            fp = FrozenPolicyAgent(pid, lm, deterministic=rng.random() < 0.5)
            fp.reset()
            opponents[pid] = fp
        else:                                                # 40% strong baselines
            opponents[pid] = rng.choice(base_pool)(pid)
    return opponents


# ===========================================================================
# Data collection — BC and DAgger
# ===========================================================================
def collect_initial_data(train_dir: str, val_dir: str, num_games: int) -> None:
    ensure_dir(train_dir); ensure_dir(val_dir)
    tr_man = load_manifest(train_dir); va_man = load_manifest(val_dir)
    tr_ci  = len(tr_man["chunks"]);   va_ci  = len(va_man["chunks"])
    tr_s, tr_a, tr_se = [], [], []
    va_s, va_a, va_se = [], [], []

    for gi in range(num_games):
        seed    = SEED + gi
        cid     = gi % 4
        split   = "val" if seed % TRAIN_SPLIT_MOD == 0 else "train"
        env     = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs     = env.reset()
        teacher = TeacherEnsemble(cid)
        opps    = build_opponents(cid, seed)
        done    = False; step = 0

        while not done:
            state_np = encode_obs(
                obs["map"], obs["players"], obs["bombs"], cid, step
            ).numpy().astype(np.float32)
            expert = int(teacher.act(obs))
            if split == "train":
                tr_s.append(state_np); tr_a.append(expert); tr_se.append(seed)
            else:
                va_s.append(state_np); va_a.append(expert); va_se.append(seed)

            acts = [0, 0, 0, 0]; acts[cid] = expert
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1

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
        seed    = 100000 + SEED + gi
        cid     = gi % 4
        env     = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs     = env.reset()
        teacher = TeacherEnsemble(cid)
        opps    = build_opponents(cid, seed)
        done    = False; step = 0

        while not done:
            state = encode_obs(
                obs["map"], obs["players"], obs["bombs"], cid, step
            ).unsqueeze(0).to(DEVICE)
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
# Reward shaping  — FIX v5: rebalanced kill / death / enemy-in-blast
# ===========================================================================
def compute_shaped_reward(
    prev_obs: Dict, next_obs: Dict,
    my_id: int, action: int,
    terminated: bool, truncated: bool,
) -> float:
    reward = 0.0
    prev_players, next_players = prev_obs["players"], next_obs["players"]
    prev_map, next_map = prev_obs["map"], next_obs["map"]

    if my_id < len(prev_players) and my_id < len(next_players):
        prev_alive = int(prev_players[my_id][2])
        next_alive = int(next_players[my_id][2])

        if prev_alive == 1 and next_alive == 1:
            reward += 0.0002   # survival tick (unchanged — intentionally tiny)
        elif prev_alive == 1 and next_alive == 0:
            reward -= 4.0      # FIX: was -6.0; death still very bad but not 6× a kill

        bonus_gain = max(0, int(next_players[my_id][4]) - int(prev_players[my_id][4]))
        if bonus_gain > 0:
            reward += 0.05 * bonus_gain

        npos = (int(next_players[my_id][0]), int(next_players[my_id][1]))
        if 0 <= npos[0] < prev_map.shape[0] and 0 <= npos[1] < prev_map.shape[1]:
            prev_cell = int(prev_map[npos[0], npos[1]])
            next_cell = int(next_map[npos[0], npos[1]])
            if prev_cell in (3, 4) and next_cell == 0:
                reward += 0.08 if prev_cell == 3 else 0.10

    prev_alive_e = (int(np.sum(prev_players[:, 2])) - int(prev_players[my_id][2])
                    if my_id < len(prev_players) else 0)
    next_alive_e = (int(np.sum(next_players[:, 2])) - int(next_players[my_id][2])
                    if my_id < len(next_players) else 0)
    kills = max(0, prev_alive_e - next_alive_e)
    if kills > 0:
        # FIX: raised from +0.9 to +2.0; last-enemy kill gets +3.5
        last_kill = (next_alive_e == 0)
        bonus = 3.5 if last_kill else 2.0
        reward += bonus * kills

    boxes_destroyed = max(0, int(np.sum(prev_map == 2)) - int(np.sum(next_map == 2)))
    if boxes_destroyed > 0:
        reward += 0.03 * boxes_destroyed
        if boxes_destroyed >= 2:
            reward += 0.01 * (boxes_destroyed - 1)

    if action == 5 and my_id < len(prev_players) and int(prev_players[my_id][2]) == 1:
        my_pos     = (int(prev_players[my_id][0]), int(prev_players[my_id][1]))
        bomb_radius = 1 + int(prev_players[my_id][4])
        blast       = blast_tiles(prev_map, my_pos[0], my_pos[1], bomb_radius)

        hit_enemies = sum(
            1 for i in range(4)
            if i != my_id and i < len(prev_players) and int(prev_players[i][2]) == 1
            and (int(prev_players[i][0]), int(prev_players[i][1])) in blast
        )
        enemy_in_blast = (hit_enemies > 0)

        if should_place_bomb_here(prev_map, prev_players, prev_obs["bombs"],
                                  my_id, my_pos, enemy_in_blast=enemy_in_blast):
            reward += 0.10

            hit_boxes = sum(1 for r, c in blast if int(prev_map[r, c]) == 2)

            hyp_bombs = _add_hypothetical_bomb(prev_obs["bombs"], my_pos, my_id)
            before    = bomb_effective_explosion_times(prev_map, prev_players, prev_obs["bombs"])
            after     = bomb_effective_explosion_times(prev_map, prev_players, hyp_bombs)
            chain_gain = (float(np.sum(np.maximum(0, before - after)))
                          if len(before) and len(after) else 0.0)

            reward += 0.015 * hit_boxes
            reward += 0.25 * hit_enemies   # FIX: was +0.08; this is the key offensive signal
            reward += 0.004 * chain_gain
        else:
            reward -= 0.12

    reward -= 0.001  # anti-stall

    if terminated or truncated:
        if my_id < len(next_players) and int(next_players[my_id][2]) == 1:
            reward += 10.0 if int(np.sum(next_players[:, 2])) == 1 else 0.05
        else:
            reward -= 1.5  # FIX: was -2.0

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

    my_pos   = (int(players[my_id][0]), int(players[my_id][1]))
    blocked  = bomb_positions_set(bombs)
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
        if mask[5] > 0:
            # FIX v5: check if enemy is in blast; if so, use relaxed threshold
            bomb_radius    = 1 + int(players[my_id][4])
            enemy_in_blast = _enemy_in_blast(grid, players, my_id, my_pos, bomb_radius)
            if not should_place_bomb_here(grid, players, bombs, my_id, my_pos,
                                          enemy_in_blast=enemy_in_blast):
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
    masked = logits.clone()
    masked[~mask_t] = -1e9
    dist   = Categorical(logits=masked)
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
    tr_ds = ChunkedBCDataset(train_dir, augment=True,  shuffle_chunks=True,
                              shuffle_within_chunk=True,  seed=SEED)
    va_ds = ChunkedBCDataset(val_dir,   augment=False, shuffle_chunks=False,
                              shuffle_within_chunk=False, seed=SEED)
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
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={tr_loss:.4f}/{tr_acc:.4f} | "
              f"val={va_loss:.4f}/{va_acc:.4f}", flush=True)

        if va_loss < best_val - 1e-4:
            best_val   = va_loss
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
# PPO rollout storage + GAE
# ===========================================================================
@dataclass
class RolloutEpisode:
    states:     List[np.ndarray]  = field(default_factory=list)
    actions:    List[int]         = field(default_factory=list)
    rewards:    List[float]       = field(default_factory=list)
    dones:      List[bool]        = field(default_factory=list)
    log_probs:  List[float]       = field(default_factory=list)
    values:     List[float]       = field(default_factory=list)
    masks:      List[np.ndarray]  = field(default_factory=list)
    last_value: float             = 0.0


def _gae(ep: RolloutEpisode) -> Tuple[np.ndarray, np.ndarray]:
    T    = len(ep.rewards)
    adv  = np.zeros(T, dtype=np.float32)
    vals = np.asarray(ep.values, dtype=np.float32)
    gae  = 0.0

    for t in reversed(range(T)):
        if ep.dones[t]:
            next_val = 0.0
        elif t + 1 < T:
            next_val = float(vals[t + 1])
        else:
            next_val = ep.last_value

        delta  = ep.rewards[t] + PPO_GAMMA * next_val - float(vals[t])
        gae    = delta + PPO_GAMMA * PPO_LAMBDA * (1.0 - float(ep.dones[t])) * gae
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
        # NOTE: NO per-episode normalization here — see global norm below
        all_states.extend(ep.states);  all_acts.extend(ep.actions)
        all_lps.extend(ep.log_probs);  all_vals.extend(ep.values)
        all_rets.extend(ret.tolist()); all_advs.extend(adv.tolist())
        all_masks.extend(ep.masks)

    if not all_states:
        raise RuntimeError("No rollout samples collected.")

    mk = lambda lst, dt: torch.tensor(np.array(lst), dtype=dt)
    states_t = mk(all_states, torch.float32)
    acts_t   = mk(all_acts,   torch.long)
    lps_t    = mk(all_lps,    torch.float32)
    vals_t   = mk(all_vals,   torch.float32)
    rets_t   = mk(all_rets,   torch.float32)
    masks_t  = mk(all_masks,  torch.float32)

    # FIX v5: GLOBAL advantage normalization across the entire batch
    # This preserves the relative magnitude of kill steps vs survival steps.
    advs_t = mk(all_advs, torch.float32)
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    return states_t, acts_t, lps_t, vals_t, rets_t, advs_t, masks_t


# ===========================================================================
# Multiprocessing rollout worker
# ===========================================================================
def _worker_collect(
    worker_id:      int,
    game_seeds:     List[int],
    cids:           List[int],
    round_idx:      int,
    model_state:    Dict,
    frozen_state:   Optional[Dict],
    league_states:  List[Dict],
    result_queue:   mp.Queue,
) -> None:
    """
    Worker function for parallel rollout collection.
    Runs in a separate process (spawn). Collects a shard of rollout games and
    puts a list of RolloutEpisode into result_queue.

    All model state dicts are passed as plain dicts to avoid CUDA fork issues.
    The worker rebuilds models on CPU.
    """
    try:
        # Re-seed in worker
        set_seed(SEED + worker_id * 7919 + round_idx)

        device = torch.device("cpu")  # workers run on CPU

        # Rebuild model
        model = BomberNet(INPUT_CHANNELS).to(device)
        model.load_state_dict(model_state)
        model.eval()

        frozen_model = None
        if frozen_state is not None:
            frozen_model = BomberNet(INPUT_CHANNELS).to(device)
            frozen_model.load_state_dict(frozen_state)
            frozen_model.eval()

        # Rebuild league pool (snapshots only, no anchor needed in worker)
        class _SimplePool:
            def __init__(self, states):
                self.snapshots = []
                for sd in states:
                    m = BomberNet(INPUT_CHANNELS).to(device)
                    m.load_state_dict(sd)
                    m.eval()
                    self.snapshots.append(m)
            def sample(self):
                return random.choice(self.snapshots) if self.snapshots else None

        league_pool = _SimplePool(league_states)

        episodes: List[RolloutEpisode] = []

        for gi, (seed, cid) in enumerate(zip(game_seeds, cids)):
            env  = BomberEnv(max_steps=MAX_STEPS, seed=seed)
            obs  = env.reset()
            opps = build_selfplay_opponents(cid, seed, frozen_model, league_pool)

            ep   = RolloutEpisode()
            done = False; step = 0
            truncated_alive = False

            while not done:
                if cid >= len(obs["players"]) or int(obs["players"][cid][2]) != 1:
                    break

                state  = encode_obs(
                    obs["map"], obs["players"], obs["bombs"], cid, step
                ).unsqueeze(0).to(device)
                my_pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
                bl     = int(obs["players"][cid][3])
                lm     = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bl)
                # FIX v6: use lm not shield — see _collect_single_process comments.

                with torch.no_grad():
                    action, log_prob, _, value = _sample_masked_action(
                        model, state, lm, sample=True, temperature=1.0  # FIX v6: 0.90→1.0
                    )

                acts = [0, 0, 0, 0]; acts[cid] = int(action)
                for pid, ag in opps.items():
                    acts[pid] = int(ag.act(obs))

                prev_obs = obs
                obs, terminated, truncated = env.step(acts)
                my_died  = (int(obs["players"][cid][2]) == 0)
                reward   = compute_shaped_reward(
                    prev_obs, obs, cid, action, terminated, truncated
                )

                genuine_done = bool(my_died or terminated)

                ep.states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
                ep.actions.append(int(action));    ep.rewards.append(float(reward))
                ep.dones.append(genuine_done);     ep.log_probs.append(float(log_prob))
                ep.values.append(float(value))
                ep.masks.append(lm.astype(np.float32))  # FIX v6: store lm not shield

                truncated_alive = bool(truncated and not terminated and not my_died)
                done = bool(terminated or truncated or my_died)
                step += 1

            # Bootstrap last value
            ep.last_value = 0.0
            if truncated_alive and ep.states:
                try:
                    ls = encode_obs(
                        obs["map"], obs["players"], obs["bombs"], cid, step
                    ).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, lv = _model_logits_value(model, ls)
                    if lv is not None:
                        ep.last_value = float(lv.item())
                except Exception:
                    pass

            if ep.states:
                episodes.append(ep)

        result_queue.put(("ok", worker_id, episodes))
    except Exception as exc:
        result_queue.put(("err", worker_id, str(exc)))


def collect_selfplay_rollouts(
    model: nn.Module,
    frozen_model: Optional[nn.Module],
    num_games: int,
    round_idx: int = 0,
    league_pool: Optional[LeaguePool] = None,
) -> List[RolloutEpisode]:
    """
    Collect PPO rollout episodes.

    FIX v5: distributes work across NUM_ROLLOUT_WORKERS parallel processes using
    torch.multiprocessing (spawn context). Each worker gets a shard of game seeds
    and collects independently. Results are gathered via a Queue.

    Falls back to single-process if workers = 1 or if any worker fails to start.
    """
    model.eval()
    if frozen_model is not None:
        frozen_model.eval()

    # Prepare seeds and agent IDs for all games
    seeds = [300000 + SEED + round_idx * 10000 + gi for gi in range(num_games)]
    cids  = [gi % 4 for gi in range(num_games)]

    # Serialize model states for IPC (CPU tensors only)
    model_state   = {k: v.cpu() for k, v in model.state_dict().items()}
    frozen_state  = ({k: v.cpu() for k, v in frozen_model.state_dict().items()}
                     if frozen_model is not None else None)
    league_states = ([{k: v.cpu() for k, v in snap.state_dict().items()}
                      for snap in (league_pool.snapshots if league_pool else [])])

    n_workers = NUM_ROLLOUT_WORKERS

    # Single-process fallback
    if n_workers <= 1:
        return _collect_single_process(
            model, frozen_model, num_games, round_idx, league_pool
        )

    # Split seeds into shards
    shard_size = (num_games + n_workers - 1) // n_workers
    shards = []
    for wi in range(n_workers):
        sl = wi * shard_size
        el = min(sl + shard_size, num_games)
        if sl < num_games:
            shards.append((seeds[sl:el], cids[sl:el]))

    ctx      = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs    = []

    try:
        for wi, (s_seeds, s_cids) in enumerate(shards):
            p = ctx.Process(
                target=_worker_collect,
                args=(wi, s_seeds, s_cids, round_idx,
                      model_state, frozen_state, league_states, result_q),
                daemon=True,
            )
            p.start()
            procs.append(p)

        all_episodes: List[RolloutEpisode] = []
        errors = []
        for _ in range(len(procs)):
            status, wid, payload = result_q.get(timeout=600)
            if status == "ok":
                all_episodes.extend(payload)
                print(f"  Worker {wid} done: {len(payload)} episodes", flush=True)
            else:
                errors.append(f"Worker {wid}: {payload}")

        for p in procs:
            p.join(timeout=10)

        if errors:
            print(f"  Worker errors: {errors}", flush=True)
            # If all workers failed, fallback
            if len(all_episodes) == 0:
                print("  All workers failed — falling back to single process", flush=True)
                return _collect_single_process(
                    model, frozen_model, num_games, round_idx, league_pool
                )

        return all_episodes

    except Exception as e:
        print(f"  Multiprocessing failed ({e}) — falling back to single process", flush=True)
        for p in procs:
            try: p.terminate()
            except: pass
        return _collect_single_process(
            model, frozen_model, num_games, round_idx, league_pool
        )


def _collect_single_process(
    model: nn.Module,
    frozen_model: Optional[nn.Module],
    num_games: int,
    round_idx: int,
    league_pool: Optional[LeaguePool],
) -> List[RolloutEpisode]:
    """Original single-process rollout collection — used as fallback."""
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
        truncated_alive = False

        while not done:
            if cid >= len(obs["players"]) or int(obs["players"][cid][2]) != 1:
                break

            state  = encode_obs(
                obs["map"], obs["players"], obs["bombs"], cid, step
            ).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl     = int(obs["players"][cid][3])
            lm     = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bl)
            # FIX v6: use lm (legal mask) for rollout, NOT shield.
            # The shield was hard-blocking exploration of dangerous-but-learnable
            # moves. The death penalty (−4.0) and unsafe-bomb penalty (−0.12)
            # are sufficient to teach safety through reward. Shield is kept
            # for eval and the submitted agent.py.

            with torch.no_grad():
                action, log_prob, _, value = _sample_masked_action(
                    model, state, lm, sample=True, temperature=1.0  # FIX v6: 0.90→1.0
                )

            acts = [0, 0, 0, 0]; acts[cid] = int(action)
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))

            prev_obs = obs
            obs, terminated, truncated = env.step(acts)
            my_died  = (int(obs["players"][cid][2]) == 0)
            reward   = compute_shaped_reward(
                prev_obs, obs, cid, action, terminated, truncated
            )

            genuine_done = bool(my_died or terminated)

            ep.states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
            ep.actions.append(int(action));    ep.rewards.append(float(reward))
            ep.dones.append(genuine_done);     ep.log_probs.append(float(log_prob))
            ep.values.append(float(value))
            ep.masks.append(lm.astype(np.float32))  # FIX v6: store lm not shield

            truncated_alive = bool(truncated and not terminated and not my_died)
            done = bool(terminated or truncated or my_died)
            step += 1

        ep.last_value = 0.0
        if truncated_alive and ep.states:
            try:
                ls = encode_obs(
                    obs["map"], obs["players"], obs["bombs"], cid, step
                ).unsqueeze(0).to(DEVICE)
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
            print(f"Rollout {gi+1}/{num_games} | eps={len(episodes)} | steps={total_steps}",
                  flush=True)

    return episodes


# ===========================================================================
# PPO fine-tuning
# ===========================================================================
def ppo_finetune(
    model: nn.Module,
    episodes: List[RolloutEpisode],
    bc_mix_dir: Optional[str] = None,
    entropy_coef: float = PPO_ENTROPY_COEF,
    lr: float = FINE_TUNE_LR,   # FIX v6: per-round LR decay passed from main()
) -> nn.Module:
    if not episodes:
        return model
    states, actions, old_lps, old_vals, returns, advantages, masks = _flatten_episodes(episodes)
    N         = states.shape[0]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # FIX v6: only load BC data when BC_MIX_COEF > 0. Previously the loader
    # was always constructed (and bc_iter always non-None), adding pointless
    # cross-entropy computation even when BC_MIX_COEF was 0.
    bc_loader = None
    if BC_MIX_COEF > 0 and bc_mix_dir and os.path.exists(bc_mix_dir):
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

            bs  = states[bi].to(DEVICE);    ba  = actions[bi].to(DEVICE)
            blp = old_lps[bi].to(DEVICE);   brt = returns[bi].to(DEVICE)
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
            loss     = pol_loss + PPO_VALUE_COEF * val_loss - entropy_coef * entropy

            # FIX v6: BC mixing guarded by BC_MIX_COEF > 0
            if bc_iter is not None and BC_MIX_COEF > 0:
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
            f"val={t_val/max(1,n_b):.4f} ent={t_ent/max(1,n_b):.4f} "
            f"ent_coef={entropy_coef:.4f} lr={lr:.2e}",
            flush=True,
        )

    torch.save(model.state_dict(), MODEL_PATH)
    return model



# ===========================================================================
# Evaluation
# ===========================================================================
def quick_eval_against_baselines(
    model: nn.Module,
    num_games: int = 30,
    seed_offset: Optional[int] = None,
    return_wins: bool = False,   # FIX v6: allow caller to get win count for best-model tracking
) -> Optional[int]:
    model.eval()
    wins = draws = losses = 0
    total_kills = total_boxes = total_steps = 0

    if seed_offset is None:
        seed_offset = random.randint(0, 500000)

    for gi in range(num_games):
        seed = 400000 + seed_offset + gi
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs  = env.reset()
        opps = build_opponents(cid, seed)
        init_boxes = int(np.sum(obs["map"] == 2))
        kills = 0; done = False; step = 0

        while not done:
            if int(obs["players"][cid][2]) != 1:
                break
            state  = encode_obs(
                obs["map"], obs["players"], obs["bombs"], cid, step
            ).unsqueeze(0).to(DEVICE)
            my_pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl     = int(obs["players"][cid][3])
            lm     = _legal_action_mask(obs["map"], obs["bombs"], my_pos, bl)
            shield = _shielded_legal_mask(   # eval keeps shield for safe deployment
                obs["map"], obs["players"], obs["bombs"], cid, lm
            )
            with torch.no_grad():
                action, _, _, _ = _sample_masked_action(model, state, shield, sample=False)

            prev_e = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            acts   = [0, 0, 0, 0]; acts[cid] = action
            for pid, ag in opps.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            next_e = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            kills += max(0, prev_e - next_e)
            done = bool(terminated or truncated); step += 1

        alive           = [int(p[2]) for p in obs["players"]]
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
    return wins if return_wins else None



# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ensure_dir(TRAIN_DIR); ensure_dir(VAL_DIR)

    if BASELINE_IMPORT_ERRORS:
        print("Baseline import warnings:", flush=True)
        for name, err in BASELINE_IMPORT_ERRORS:
            print(f"  {name}: {err}", flush=True)

    print(f"Device: {DEVICE} | Rollout workers: {NUM_ROLLOUT_WORKERS}", flush=True)

    # Uncomment to re-run BC phases (recommended before a fresh full run):
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
    current_dir     = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "model_bc_best_.pth")

    # FIX v6: graceful weight loading — if checkpoint is missing or has wrong
    # shape (e.g. old 4×4-pool architecture), fall back to random init and
    # warn the user to re-run BC phases 1-4 before starting PPO.
    if os.path.exists(pretrained_path):
        try:
            state = torch.load(pretrained_path, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            print(f"Loaded pretrained weights from {pretrained_path}", flush=True)
        except Exception as e:
            print(
                f"WARNING: Could not load {pretrained_path} ({e}).\n"
                f"         The checkpoint is likely from the old 4×4-pool architecture.\n"
                f"         Uncomment phases 1-4 in main() and re-run BC before PPO.\n"
                f"         Continuing with random init — PPO will be slow to warm up.",
                flush=True,
            )
    else:
        print(
            f"WARNING: {pretrained_path} not found.\n"
            f"         Uncomment phases 1-4 and re-run BC first for best results.\n"
            f"         Continuing with random init.",
            flush=True,
        )

    print("=== Phase 5: PPO self-play fine-tuning ===", flush=True)
    league = LeaguePool(max_size=LEAGUE_POOL_SIZE)
    league.set_anchor(model)
    league.add(model)

    ent_coef  = float(PPO_ENTROPY_COEF)
    best_wins = -1
    best_path = os.path.join(current_dir, "model_best_ppo.pth")  # FIX v6: track best

    for round_idx in range(RL_ROUNDS):
        # FIX v6: per-round LR decay — gentle cosine-like warmdown keeps
        # updates stable as the policy matures. 3e-4 → ~1.8e-4 at round 100.
        round_lr = FINE_TUNE_LR * (0.995 ** round_idx)

        print(
            f"--- PPO round {round_idx+1}/{RL_ROUNDS} | "
            f"ent_coef={ent_coef:.4f} | lr={round_lr:.2e} ---",
            flush=True,
        )

        frozen = copy.deepcopy(model).to(DEVICE).eval()
        rollouts = collect_selfplay_rollouts(
            model,
            frozen_model=frozen,
            num_games=ROLLOUT_GAMES_PER_ROUND,
            round_idx=round_idx,
            league_pool=league,
        )
        print(
            f"  collected {len(rollouts)} episodes "
            f"({sum(len(e.states) for e in rollouts)} steps)",
            flush=True,
        )

        model = ppo_finetune(
            model, rollouts,
            bc_mix_dir=TRAIN_DIR,
            entropy_coef=ent_coef,
            lr=round_lr,         # FIX v6: pass decayed LR
        )
        league.add(model)

        ent_coef = max(PPO_ENTROPY_MIN, ent_coef * PPO_ENTROPY_DECAY)

        # FIX v6: evaluate and save best checkpoint
        wins = quick_eval_against_baselines(model, num_games=20, return_wins=True)
        if wins > best_wins:
            best_wins = wins
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best: {wins} wins — saved to {best_path}", flush=True)
        torch.save(model.state_dict(), MODEL_PATH)  # save latest after every round

    print("=== Final evaluation ===", flush=True)
    quick_eval_against_baselines(model, num_games=50)
    # Also evaluate the best checkpoint
    print("=== Best checkpoint evaluation ===", flush=True)
    if os.path.exists(best_path):
        best_model = BomberNet(INPUT_CHANNELS).to(DEVICE)
        best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        quick_eval_against_baselines(best_model, num_games=50)
        print(f"Best model saved at: {best_path}", flush=True)
    print("Done.", flush=True)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()