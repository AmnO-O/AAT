"""
ClaudFat_v7.py — Pure self-play PPO from random initialization.

No BC, no DAgger. The agent learns entirely through self-play.

Key design decisions:

  [MAPS]      100 fixed training maps, reused every round. Same maps across
              all rounds means the value function calibrates precisely over
              time — inspired by chess/Go where the starting position is
              always identical. Opponent seeds vary per round so each map
              is played against different opponents each time.

  [EVAL]      50 fixed held-out maps, NEVER used in training.
              Every eval call uses the same 50 maps → stable, comparable
              signal across rounds. Not a moving target.

  [REWARD]    kill + die = 0 (risk-neutral on kills).
              kill=3.5, step_death=-3.5, terminal_death=removed.
              Win (+4.5 last_kill + 10.0 terminal) = 14.5 dominant signal.
              The value function naturally learns dying forfeits future wins.

  [CURRICULUM] 4-phase opponent schedule:
              Phase 0 (R0-9):   80% self/league, 20% weak → easy wins to bootstrap
              Phase 1 (R10-29): 65% self/league, 35% weak+medium → growing challenge
              Phase 2 (R30-59): 50% self/league, 50% medium+strong → real competition
              Phase 3 (R60+):   35% self/league, 65% strong → tournament-level

  [OPTIMIZER] Persistent AdamW across rounds. Momentum accumulates = better
              gradient direction estimates over time. Fixed LR 3e-4.

  [ARCH]      BomberNet v6: width=64, pool=7×7, AlphaZero-style 1×1-conv heads.
              Value path detached from backbone → backbone trains only on
              policy signal, not corrupted by 160× stronger value gradients.
              375K params total.

  [BUG FIXED] Previous "fixed maps" used seed=CONST for ALL 600 games per
              round → all 600 games were identical → catastrophic overfitting
              to ONE map → 0 wins on eval. Fixed: 100 distinct map seeds,
              each played 6 times per round.
"""

import copy
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append(os.getcwd())
from engine.game import BomberEnv

# ---------------------------------------------------------------------------
# Baseline imports (graceful fallback if missing)
# ---------------------------------------------------------------------------
def _try_import(module: str, cls: str):
    try:
        return getattr(__import__(module, fromlist=[cls]), cls)
    except Exception:
        return None

TacticalRuleAgent = _try_import("agent.tactical_rule_agent", "TacticalRuleAgent")
GeniusRuleAgent   = _try_import("agent.genius_rule_agent",   "GeniusRuleAgent")
SmarterRuleAgent  = _try_import("agent.smarter_rule_agent",  "SmarterRuleAgent")
BoxFarmerAgent    = _try_import("agent.box_farmer_agent",    "BoxFarmerAgent")
SimpleRuleAgent   = _try_import("agent.simple_rule_agent",   "SimpleRuleAgent")
RandomAgent       = _try_import("agent.random_agent",        "RandomAgent")

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

SPATIAL_CHANNELS: List[int] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,21,24,25,26]  # 20
SCALAR_CHANNELS:  List[int] = [14,17,18,19,20,22,23]                                  # 7

# ---------------------------------------------------------------------------
# Fixed map pools
# Training maps: 100 distinct seeds, reused every round.
# Eval maps:     50 distinct seeds, NEVER used during training.
# ---------------------------------------------------------------------------
N_TRAIN_MAPS     = 25
N_EVAL_MAPS      = 50
_TRAIN_MAP_SEEDS = [300_000 + SEED + i * 137 for i in range(N_TRAIN_MAPS)]
_EVAL_MAP_SEEDS  = [900_000 + SEED + i * 137 for i in range(N_EVAL_MAPS)]

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
RL_ROUNDS               = 150
ROLLOUT_GAMES_PER_ROUND = 400
PPO_EPOCHS              = 3
PPO_BATCH_SIZE          = 256
PPO_CLIP_EPS            = 0.20
PPO_GAMMA               = 0.98
PPO_LAMBDA              = 0.95
PPO_VALUE_COEF          = 0.5
PPO_ENTROPY_COEF        = 0.03
PPO_ENTROPY_DECAY       = 0.98   # slow decay — 150 rounds, need exploration longer
PPO_ENTROPY_MIN         = 0.005
PPO_MAX_GRAD_NORM       = 0.5
LEAGUE_POOL_SIZE        = 10
FINE_TUNE_LR            = 3e-4
WEIGHT_DECAY            = 1e-4

MODEL_PATH    = "model_bomber.pth"
BEST_PPO_PATH = "model_bomber_best.pth"

# ===========================================================================
# Seeding
# ===========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(SEED)
if hasattr(torch, "set_float32_matmul_precision"):
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# ===========================================================================
# Board / movement helpers
# ===========================================================================
MOVES = {0: (0,0), 1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}

def next_pos(pos: Tuple[int,int], action: int) -> Tuple[int,int]:
    dr, dc = MOVES[int(action)]; return pos[0]+dr, pos[1]+dc

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)

def bomb_positions_set(bombs: np.ndarray) -> set:
    if bombs is None or len(bombs) == 0: return set()
    return {(int(b[0]), int(b[1])) for b in bombs}

def bomb_radius_for_owner(players: np.ndarray, owner: int) -> int:
    if 0 <= owner < len(players) and int(players[owner][2]) == 1:
        return 1 + int(players[owner][4])
    return 1

def blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set:
    tiles = {(bx, by)}
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1, radius+1):
            r, c = bx+dr*d, by+dc*d
            if not in_bounds(r, c): break
            cell = int(grid[r, c])
            if cell == 1: break
            tiles.add((r, c))
            if cell == 2: break
    return tiles

def blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    mask[bx, by] = True
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1, radius+1):
            r, c = bx+dr*d, by+dc*d
            if not in_bounds(r, c): break
            cell = int(grid[r, c])
            if cell == 1: break
            mask[r, c] = True
            if cell == 2: break
    return mask

def bomb_effective_explosion_times(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    if bombs is None or len(bombs) == 0: return np.zeros((0,), dtype=np.int32)
    n = len(bombs)
    times = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    blasts = []
    for i in range(n):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        blasts.append(blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), bomb_radius_for_owner(players, owner)))
    q = deque(range(n)); in_q = [True]*n
    while q:
        i = q.popleft(); in_q[i] = False; ti = int(times[i])
        for j in range(n):
            if i == j: continue
            if (int(bombs[j][0]), int(bombs[j][1])) in blasts[i] and int(times[j]) > ti:
                times[j] = ti
                if not in_q[j]: q.append(j); in_q[j] = True
    return times

# ===========================================================================
# Danger / explosion planes
# ===========================================================================
def explosion_time_plane(grid, players, bombs, horizon=EXPLOSION_TIME_HORIZON):
    plane = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0: return plane
    times = bomb_effective_explosion_times(grid, players, bombs)
    denom = horizon if horizon > 0 else 1.0
    for i in range(len(bombs)):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        norm_t = min(float(max(0, int(times[i]))), horizon) / denom
        bm = blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), radius)
        plane[bm] = np.minimum(plane[bm], norm_t)
    return plane

def danger_plane(grid, players, bombs, timer_threshold=1):
    if bombs is None or len(bombs) == 0: return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    plane = explosion_time_plane(grid, players, bombs)
    thr = float(timer_threshold) / EXPLOSION_TIME_HORIZON
    return (plane <= thr).astype(np.float32)

def immediate_danger_plane(grid, players, bombs): return danger_plane(grid, players, bombs, 1)

def chain_danger_plane(grid, players, bombs, chain_horizon=3):
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0: return plane
    original  = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    effective = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        eff, orig = int(effective[i]), int(original[i])
        if eff <= 1 or eff > chain_horizon or eff >= orig: continue
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        plane[blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), bomb_radius_for_owner(players, owner))] = 1.0
    return plane

def future_danger_plane(grid, players, bombs, horizon=EXPLOSION_TIME_HORIZON):
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0: return plane
    effective = bomb_effective_explosion_times(grid, players, bombs)
    denom = float(max(1.0, horizon))
    for i in range(len(bombs)):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        t = float(max(0, int(effective[i])))
        score = 1.0 - min(t, denom)/denom
        if score <= 0: continue
        plane[blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), bomb_radius_for_owner(players, owner))] = np.maximum(
            plane[blast_mask(grid, int(bombs[i][0]), int(bombs[i][1]), bomb_radius_for_owner(players, owner))], score)
    return plane

def tile_earliest_explosion_times(grid, players, bombs):
    times = np.full((BOARD_SIZE, BOARD_SIZE), 9999, dtype=np.int32)
    if bombs is None or len(bombs) == 0: return times
    eff = bomb_effective_explosion_times(grid, players, bombs)
    for i, b in enumerate(bombs):
        owner = int(b[3]) if bombs.shape[1] > 3 else -1
        bm = blast_mask(grid, int(b[0]), int(b[1]), bomb_radius_for_owner(players, owner))
        times[bm] = np.minimum(times[bm], int(max(0, eff[i])))
    return times

def bomb_pressure_plane(grid, players, bombs, my_id):
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None: bombs = np.zeros((0,4), dtype=np.int8)
    for pid in range(4):
        if pid==my_id or pid>=len(players) or int(players[pid][2])!=1: continue
        if int(players[pid][3]) <= 0: continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c): continue
        plane[blast_mask(grid, r, c, 1+int(players[pid][4]))] = 1.0
    return plane

def future_bomb_pressure_plane(grid, players, bombs, my_id):
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None: bombs = np.zeros((0,4), dtype=np.int8)
    blocked = bomb_positions_set(bombs)
    for pid in range(4):
        if pid==my_id or pid>=len(players) or int(players[pid][2])!=1: continue
        if int(players[pid][3]) <= 0: continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c): continue
        radius = 1+int(players[pid][4])
        candidates = [(r, c)]
        for a in (1,2,3,4):
            nr, nc = next_pos((r,c), a)
            if passable(grid, nr, nc) and (nr,nc) not in blocked:
                candidates.append((nr, nc))
        for pr, pc in candidates:
            bm = blast_mask(grid, pr, pc, radius)
            plane[bm] = np.maximum(plane[bm], 0.5)
    return plane

def bottleneck_risk_plane(grid, players, bombs, my_id):
    """Vectorized: scores each tile by how trapped it is."""
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1: return plane
    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    blocked = bomb_positions_set(bombs)

    # Vectorized passable mask
    pass_v = np.isin(grid, [0, 3, 4]).copy()
    for br, bc in blocked:
        if in_bounds(br, bc): pass_v[br, bc] = False

    # Count exits via numpy shifts
    def _shift(arr, dr, dc):
        out = np.zeros_like(arr)
        if dr == -1: out[:-1, :] = arr[1:, :]
        elif dr == 1: out[1:, :]  = arr[:-1, :]
        elif dc == -1: out[:, :-1] = arr[:, 1:]
        elif dc == 1:  out[:, 1:]  = arr[:, :-1]
        return out

    exits = sum(_shift(pass_v.astype(np.int32), dr, dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])

    exp_times  = tile_earliest_explosion_times(grid, players, bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
    dangerous  = (danger_now > 0) | (exp_times <= 2)
    fragile = sum(_shift((dangerous & pass_v).astype(np.int32), dr, dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])

    plane = np.where(exits == 0,   1.00, plane)
    plane = np.where((exits==1)&(fragile>=1), 0.85, plane)
    plane = np.where((exits==1)&(fragile==0), 0.65, plane)
    plane = np.where((exits==2)&(fragile>=2), 0.40, plane)
    plane = np.where((exits==2)&(fragile< 2), 0.20, plane)
    plane = plane * pass_v

    row_idx = np.arange(BOARD_SIZE)[:,None]; col_idx = np.arange(BOARD_SIZE)[None,:]
    manhattan = np.abs(row_idx - my_r) + np.abs(col_idx - my_c)
    plane = np.maximum(plane, np.where((manhattan<=1)&pass_v, 0.75, 0.0))
    plane = np.maximum(plane, np.where((manhattan<=2)&pass_v, 0.35, 0.0))
    return plane.astype(np.float32)

# ===========================================================================
# BFS / escape utilities
# ===========================================================================
def escape_margin_from_position(grid, players, bombs, start, max_depth=6):
    exp_times = tile_earliest_explosion_times(grid, players, bombs)
    blocked = bomb_positions_set(bombs)
    q = deque([(start, 0)]); seen = {start}; best = -9999
    while q:
        pos, dist = q.popleft()
        margin = int(exp_times[pos[0], pos[1]]) - dist
        if margin > best: best = margin
        if dist >= max_depth: continue
        for a in (1,2,3,4):
            npos = next_pos(pos, a)
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]): continue
            seen.add(npos); q.append((npos, dist+1))
    return -1.0 if best < -1000 else float(best)

def time_safe_escape_score(grid, players, bombs, my_id):
    if my_id >= len(players) or int(players[my_id][2]) != 1: return 0.0
    pos = (int(players[my_id][0]), int(players[my_id][1]))
    m = escape_margin_from_position(grid, players, bombs, pos)
    return float(np.clip(m/6.0, 0.0, 1.0)) if m > 0 else 0.0

def bfs_distance_to_targets(grid, start, targets, bombs, max_depth=64):
    if not targets: return None
    blocked = bomb_positions_set(bombs)
    q = deque([(start, 0)]); seen = {start}
    while q:
        pos, dist = q.popleft()
        if pos in targets: return dist
        if dist >= max_depth: continue
        for a in (1,2,3,4):
            npos = next_pos(pos, a)
            if npos in seen or npos in blocked or not passable(grid, npos[0], npos[1]): continue
            seen.add(npos); q.append((npos, dist+1))
    return None

def bfs_reachable_count(grid, start, bombs, max_depth=3):
    blocked = bomb_positions_set(bombs)
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    visited[start[0], start[1]] = True
    q = deque([(start, 0)]); count = 0
    while q:
        pos, dist = q.popleft()
        if dist > 0: count += 1
        if dist >= max_depth: continue
        for a in (1,2,3,4):
            npos = next_pos(pos, a)
            if not in_bounds(npos[0], npos[1]): continue
            if visited[npos[0], npos[1]] or npos in blocked or not passable(grid, npos[0], npos[1]): continue
            visited[npos[0], npos[1]] = True; q.append((npos, dist+1))
    return count

def norm_dist(d, cap=24.0): return 1.0 if d is None else float(min(d, cap))/cap
def norm_scalar(x, denom): return float(np.clip(x/denom, 0.0, 1.0)) if denom > 0 else 0.0

def legal_actions(grid, bombs, my_pos, bombs_left):
    moves = [0]; blocked = bomb_positions_set(bombs)
    for a in (1,2,3,4):
        nr, nc = next_pos(my_pos, a)
        if passable(grid, nr, nc) and (nr,nc) not in blocked: moves.append(a)
    if bombs_left > 0 and my_pos not in blocked: moves.append(5)
    return moves

# ===========================================================================
# Bomb safety helpers
# ===========================================================================
def _add_hypothetical_bomb(bombs, pos, owner, timer=7):
    row = np.array([[pos[0], pos[1], timer, owner]], dtype=np.int8)
    return np.concatenate([bombs, row], axis=0) if bombs is not None and len(bombs) > 0 else row

def should_place_bomb_here(grid, players, bombs, my_id, pos, enemy_in_blast=False):
    if my_id >= len(players) or int(players[my_id][2]) != 1: return False
    if not passable(grid, pos[0], pos[1]): return False
    my_radius = 1 + int(players[my_id][4])
    hyp = _add_hypothetical_bomb(bombs, pos, my_id)
    blast = blast_tiles(grid, pos[0], pos[1], my_radius)
    blocked = bomb_positions_set(hyp)
    thr = -1.0 if enemy_in_blast else 0.0
    for a in (1,2,3,4):
        nr, nc = next_pos(pos, a)
        if not passable(grid, nr, nc) or (nr,nc) in blocked or (nr,nc) in blast: continue
        if escape_margin_from_position(grid, players, hyp, (nr,nc)) > thr: return True
    return False

def _enemy_in_blast(grid, players, my_id, pos, radius):
    blast = blast_tiles(grid, pos[0], pos[1], radius)
    for i in range(4):
        if i==my_id or i>=len(players) or int(players[i][2])!=1: continue
        if (int(players[i][0]), int(players[i][1])) in blast: return True
    return False

def safe_to_bomb_plane(grid, players, bombs, my_id):
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1: return plane
    r, c = int(players[my_id][0]), int(players[my_id][1])
    if not in_bounds(r, c) or (r,c) in bomb_positions_set(bombs): return plane
    radius = 1+int(players[my_id][4])
    blast = blast_tiles(grid, r, c, radius)
    enemies = {(int(players[i][0]),int(players[i][1])) for i in range(4) if i!=my_id and i<len(players) and int(players[i][2])==1}
    if not any(int(grid[x,y])==2 for x,y in blast) and not any(p in enemies for p in blast): return plane
    hyp = _add_hypothetical_bomb(bombs, (r,c), my_id)
    blocked_hyp = bomb_positions_set(hyp)
    thr = -1.0 if any(p in enemies for p in blast) else 0.0
    for a in (1,2,3,4):
        nr, nc = next_pos((r,c), a)
        if not passable(grid, nr, nc) or (nr,nc) in blocked_hyp or (nr,nc) in blast: continue
        if escape_margin_from_position(grid, players, hyp, (nr,nc)) > thr:
            plane[r, c] = 1.0; break
    return plane

# ===========================================================================
# Observation encoding (27 channels, unchanged)
# ===========================================================================
def encode_obs(grid, players, bombs, my_id, step):
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0] = (grid==1).astype(np.float32)
    state[1] = (grid==2).astype(np.float32)
    state[2] = (grid==0).astype(np.float32)
    state[3] = (grid==3).astype(np.float32)
    state[4] = (grid==4).astype(np.float32)
    for pid in range(4):
        if pid<len(players) and int(players[pid][2])==1:
            r,c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r,c): state[5+pid, r, c] = 1.0
    state[9]  = explosion_time_plane(grid, players, bombs)
    state[10] = immediate_danger_plane(grid, players, bombs)
    state[11] = chain_danger_plane(grid, players, bombs)
    state[12] = future_danger_plane(grid, players, bombs)
    me_alive = 0; my_pos = (0,0); bombs_left = 0
    if my_id<len(players) and int(players[my_id][2])==1:
        me_alive = 1
        mr, mc = int(players[my_id][0]), int(players[my_id][1])
        my_pos = (mr, mc)
        if in_bounds(mr,mc): state[13, mr, mc] = 1.0
        bombs_left = int(players[my_id][3])
    state[14].fill(norm_scalar(bombs_left, 5.0))
    if bombs is not None and len(bombs) > 0:
        eff = bomb_effective_explosion_times(grid, players, bombs)
        for i in range(len(bombs)):
            r, c = int(bombs[i][0]), int(bombs[i][1])
            if not in_bounds(r,c): continue
            t = max(int(eff[i]), 1)
            state[15,r,c] = max(state[15,r,c], 1.0/t)
            owner = int(bombs[i][3]) if bombs.shape[1]>3 else -1
            state[16,r,c] = max(state[16,r,c], norm_scalar(bomb_radius_for_owner(players,owner), 6.0))
    if me_alive:
        item_pos  = {(int(r),int(c)) for r,c in np.argwhere((grid==3)|(grid==4))}
        enemy_pos = {(int(players[i][0]),int(players[i][1])) for i in range(4)
                     if i!=my_id and i<len(players) and int(players[i][2])==1}
        state[17].fill(norm_dist(bfs_distance_to_targets(grid, my_pos, item_pos, bombs)))
        state[18].fill(norm_dist(bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs)))
        state[19].fill(norm_scalar(bfs_reachable_count(grid, my_pos, bombs, 3), 20.0))
        state[20].fill(time_safe_escape_score(grid, players, bombs, my_id))
        state[21] = safe_to_bomb_plane(grid, players, bombs, my_id)
    else:
        state[17].fill(1.0); state[18].fill(1.0)
    state[22].fill(norm_scalar(len(bombs) if bombs is not None else 0, 10.0))
    state[23].fill(norm_scalar(step, float(MAX_STEPS)))
    state[24] = bomb_pressure_plane(grid, players, bombs, my_id)
    state[25] = future_bomb_pressure_plane(grid, players, bombs, my_id)
    state[26] = bottleneck_risk_plane(grid, players, bombs, my_id)
    return torch.from_numpy(state)

# ===========================================================================
# Fallback rule agent (used when baseline imports fail)
# ===========================================================================
class _FallbackRuleAgent:
    def __init__(self, agent_id): self.agent_id = int(agent_id)
    def act(self, obs):
        grid, players, bombs = obs["map"], obs["players"], obs["bombs"]
        if self.agent_id>=len(players) or int(players[self.agent_id][2])!=1: return 0
        r,c = int(players[self.agent_id][0]), int(players[self.agent_id][1])
        bl = int(players[self.agent_id][3])
        dng = danger_plane(grid, players, bombs, 1)
        if dng[r,c]>0:
            moves = [a for a in (1,2,3,4) if passable(grid,*next_pos((r,c),a))
                     and dng[next_pos((r,c),a)[0], next_pos((r,c),a)[1]]==0
                     and next_pos((r,c),a) not in bomb_positions_set(bombs)]
            return int(random.choice(moves)) if moves else 0
        items = {(int(x),int(y)) for x,y in np.argwhere((grid==3)|(grid==4))}
        if items:
            best, best_d = 0, 1e9
            for a in (1,2,3,4):
                nr,nc = next_pos((r,c),a)
                if passable(grid,nr,nc) and (nr,nc) not in bomb_positions_set(bombs):
                    d = min(abs(nr-ir)+abs(nc-ic) for ir,ic in items)
                    if d < best_d: best_d,best = d,a
            if best: return int(best)
        return 5 if bl>0 else 0

# ===========================================================================
# Model: BomberNet v7 (identical architecture to v6 — COPY TO agent.py)
# ===========================================================================
_HEAD_CONV_CH = 8  # 8×7×7 + 7 scalars = 399 feat_dim

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout)
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        return torch.relu(self.bn2(self.conv2(h)) + x)

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
    _POOL    = 7                  # spatial pooling grid size

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

def _fwd(model, states): return model(states)

# ===========================================================================
# Action masking
# ===========================================================================
def _legal_mask(grid, bombs, pos, bombs_left):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in legal_actions(grid, bombs, pos, bombs_left): mask[a] = 1.0
    if mask.sum() <= 0: mask[0] = 1.0
    return mask

def _shield_mask(grid, players, bombs, my_id, lm):
    mask = lm.copy()
    if my_id>=len(players) or int(players[my_id][2])!=1:
        if mask.sum()<=0: mask[0]=1.0; return mask
    pos = (int(players[my_id][0]), int(players[my_id][1]))
    blocked = bomb_positions_set(bombs)
    dng = danger_plane(grid, players, bombs, 1)
    dng2 = danger_plane(grid, players, bombs, 2)
    in_danger = bool(dng[pos[0],pos[1]]>0 or dng2[pos[0],pos[1]]>0)
    if in_danger:
        safe = []
        for a in (1,2,3,4):
            if mask[a]<=0: continue
            nr,nc = next_pos(pos,a)
            if not passable(grid,nr,nc) or (nr,nc) in blocked: mask[a]=0.0; continue
            if escape_margin_from_position(grid,players,bombs,(nr,nc))>0: safe.append(a)
            else: mask[a]=0.0
        if safe: mask[0]=0.0
        elif mask[0]<=0: mask[0]=1.0
    else:
        if mask[5]>0:
            eib = _enemy_in_blast(grid, players, my_id, pos, 1+int(players[my_id][4]))
            if not should_place_bomb_here(grid, players, bombs, my_id, pos, eib): mask[5]=0.0
    if mask.sum()<=0: mask[0]=1.0
    return mask

def _sample_action(model, state, mask, sample=True, temperature=1.0):
    logits, value = _fwd(model, state)
    logits = logits / max(float(temperature), 1e-6)
    mt = torch.tensor(mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
    ml = logits.clone(); ml[~mt] = -1e9
    dist   = Categorical(logits=ml)
    action = dist.sample() if sample else torch.argmax(ml, -1)
    return int(action.item()), float(dist.log_prob(action).item()), float(dist.entropy().item()), float(value.item())

# ===========================================================================
# Opponent building
# ===========================================================================
def _build_pool(clss_weights):
    pool = []
    for cls, w in clss_weights:
        if cls is not None: pool.extend([cls]*w)
    return pool or [_FallbackRuleAgent]

_POOL_STRONG = _build_pool([(TacticalRuleAgent,4),(GeniusRuleAgent,3), (SmarterRuleAgent,1)])
_POOL_MEDIUM = _build_pool([(SmarterRuleAgent,1), (SimpleRuleAgent,6)])
_POOL_WEAK   = _build_pool([(SimpleRuleAgent,3),(BoxFarmerAgent,2),(_FallbackRuleAgent,1)])

def build_eval_opponents(controlled_id, seed):
    """Mixed pool for eval — all baselines equally represented."""
    rng  = random.Random(seed)
    pool = _build_pool([(TacticalRuleAgent,2),(GeniusRuleAgent,2),(SmarterRuleAgent,2),
                        (BoxFarmerAgent,1),(SimpleRuleAgent,1)])
    return {pid: rng.choice(pool)(pid) for pid in range(4) if pid!=controlled_id}

class FrozenPolicyAgent:
    def __init__(self, agent_id, model, deterministic=True):
        self.agent_id = int(agent_id); self.model = model
        self.deterministic = bool(deterministic); self._step = 0
    def reset(self): self._step = 0
    def act(self, obs):
        if self.agent_id>=len(obs["players"]) or int(obs["players"][self.agent_id][2])!=1:
            self._step += 1; return 0
        step = self._step; self._step += 1
        dev = next(self.model.parameters()).device
        state = encode_obs(obs["map"],obs["players"],obs["bombs"],self.agent_id,step).unsqueeze(0).to(dev)
        pos = (int(obs["players"][self.agent_id][0]), int(obs["players"][self.agent_id][1]))
        bl  = int(obs["players"][self.agent_id][3])
        lm  = _legal_mask(obs["map"], obs["bombs"], pos, bl)
        sm  = _shield_mask(obs["map"], obs["players"], obs["bombs"], self.agent_id, lm)
        with torch.no_grad():
            a, _, _, _ = _sample_action(self.model, state, sm, sample=not self.deterministic)
        return a

class LeaguePool:
    def __init__(self, max_size=LEAGUE_POOL_SIZE):
        self.max_size = max_size; self.snapshots: List[nn.Module] = []
    def add(self, model):
        snap = copy.deepcopy(model).cpu().eval()
        self.snapshots.append(snap)
        if len(self.snapshots) > self.max_size: self.snapshots.pop(0)
    def sample(self): return random.choice(self.snapshots) if self.snapshots else None
def build_train_opponents(controlled_id, opp_seed, frozen_model, league_pool, round_idx):
    """
    Curriculum opponent schedule.
    Phase 0 (R0-9):   40% frozen, 25% league, 15% weak, 15% medium, 5% strong   → ~65% self/league, easy wins
    Phase 1 (R10-29): 30% frozen, 20% league, 15% weak, 20% medium, 15% strong  → growing challenge
    Phase 2 (R30-59): 20% frozen, 15% league, 10% weak, 25% medium, 30% strong  → real competition
    Phase 3 (R60+):   15% frozen, 10% league,  5% weak, 20% medium, 50% strong  → tournament level
    """
    rng = random.Random(opp_seed)

    if round_idx < 10:
        p_frozen=0.40; p_league=0.25; p_weak=0.15; p_medium=0.15; p_strong=0.05
    elif round_idx < 30:
        p_frozen=0.30; p_league=0.20; p_weak=0.15; p_medium=0.20; p_strong=0.15
    elif round_idx < 60:
        p_frozen=0.20; p_league=0.15; p_weak=0.10; p_medium=0.25; p_strong=0.30
    else:
        p_frozen=0.15; p_league=0.10; p_weak=0.05; p_medium=0.20; p_strong=0.50

    opponents = {}
    for pid in [p for p in range(4) if p != controlled_id]:
        r = rng.random()
        cum_frozen = p_frozen
        cum_league = p_frozen + p_league
        cum_weak   = p_frozen + p_league + p_weak
        cum_medium = p_frozen + p_league + p_weak + p_medium
        # remaining goes to strong

        if r < cum_frozen and frozen_model is not None:
            fp = FrozenPolicyAgent(pid, frozen_model, deterministic=rng.random() < 0.6)
            fp.reset()
            opponents[pid] = fp
        elif r < cum_league and league_pool is not None and league_pool.snapshots:
            lm = league_pool.sample()
            fp = FrozenPolicyAgent(pid, lm, deterministic=rng.random() < 0.5)
            fp.reset()
            opponents[pid] = fp
        elif r < cum_weak:
            opponents[pid] = rng.choice(_POOL_WEAK)(pid)
        elif r < cum_medium:
            opponents[pid] = rng.choice(_POOL_MEDIUM)(pid)
        else:
            opponents[pid] = rng.choice(_POOL_STRONG)(pid)

    return opponents

# ===========================================================================
# Reward — kill + die = 0
# ===========================================================================
def compute_shaped_reward(prev_obs, next_obs, my_id, action, terminated, truncated):
    reward = 0.0
    pp, np_ = prev_obs["players"], next_obs["players"]
    pm, nm  = prev_obs["map"],     next_obs["map"]

    if my_id < len(pp) and my_id < len(np_):
        pa, na = int(pp[my_id][2]), int(np_[my_id][2])
        if pa==1 and na==1:
            reward += 0.0002                     # tiny survival tick
        elif pa==1 and na==0:
            reward -= 3.5                        # step-death penalty
            # NO terminal-death penalty → kill+die = 3.5-3.5 = 0
        if pa==1 and na==1:
            bonus = max(0, int(np_[my_id][4]) - int(pp[my_id][4]))
            if bonus > 0: reward += 0.05 * bonus
            npos = (int(np_[my_id][0]), int(np_[my_id][1]))
            if in_bounds(npos[0], npos[1]):
                pc = int(pm[npos[0],npos[1]]); nc = int(nm[npos[0],npos[1]])
                if pc in (3,4) and nc==0: reward += 0.08 if pc==3 else 0.10

    pe = int(np.sum(pp[:,2])) - int(pp[my_id][2]) if my_id<len(pp) else 0
    ne = int(np.sum(np_[:,2])) - int(np_[my_id][2]) if my_id<len(np_) else 0
    kills = max(0, pe - ne)
    if kills > 0:
        last = (ne == 0)
        reward += (4.5 if last else 3.5) * kills   # last kill bigger to drive win

    boxes = max(0, int(np.sum(pm==2)) - int(np.sum(nm==2)))
    if boxes > 0: reward += 0.02 * boxes + (0.01*(boxes-1) if boxes>=2 else 0)

    if action==5 and my_id<len(pp) and int(pp[my_id][2])==1:
        pos = (int(pp[my_id][0]), int(pp[my_id][1]))
        radius = 1 + int(pp[my_id][4])
        blast  = blast_tiles(pm, pos[0], pos[1], radius)
        hit_e  = sum(1 for i in range(4) if i!=my_id and i<len(pp) and int(pp[i][2])==1
                     and (int(pp[i][0]),int(pp[i][1])) in blast)
        eib = hit_e > 0
        if should_place_bomb_here(pm, pp, prev_obs["bombs"], my_id, pos, eib):
            reward += 0.08
            reward += 0.25 * hit_e
            reward += 0.015 * sum(1 for r,c in blast if int(pm[r,c])==2)
            hyp = _add_hypothetical_bomb(prev_obs["bombs"], pos, my_id)
            bef = bomb_effective_explosion_times(pm, pp, prev_obs["bombs"])
            aft = bomb_effective_explosion_times(pm, pp, hyp)
            if len(bef) and len(aft):
                reward += 0.004 * float(np.sum(np.maximum(0, bef-aft)))
        else:
            reward -= 0.12

    reward -= 0.001  # anti-stall

    if terminated or truncated:
        if my_id<len(np_) and int(np_[my_id][2])==1:
            reward += 15.0 if int(np.sum(np_[:,2]))==1 else 0.1
        # No penalty for dying at terminal: step_death already captures it

    return float(np.clip(reward, -6.0, 15.0))

# ===========================================================================
# Rollout collection — fixed training maps
# ===========================================================================
@dataclass
class Episode:
    states:    List[np.ndarray] = field(default_factory=list)
    actions:   List[int]        = field(default_factory=list)
    rewards:   List[float]      = field(default_factory=list)
    dones:     List[bool]       = field(default_factory=list)
    log_probs: List[float]      = field(default_factory=list)
    values:    List[float]      = field(default_factory=list)
    masks:     List[np.ndarray] = field(default_factory=list)
    last_val:  float            = 0.0

def _gae(ep: Episode):
    T = len(ep.rewards)
    adv = np.zeros(T, dtype=np.float32)
    vals = np.array(ep.values, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv = 0.0 if ep.dones[t] else (vals[t+1] if t+1<T else ep.last_val)
        delta = ep.rewards[t] + PPO_GAMMA*nv - vals[t]
        gae   = delta + PPO_GAMMA*PPO_LAMBDA*(1.0-float(ep.dones[t]))*gae
        adv[t] = gae
    return adv, adv + vals

def _flatten(episodes: List[Episode]):
    S,A,LP,V,R,ADV,M = [],[],[],[],[],[],[]
    for ep in episodes:
        if not ep.states: continue
        adv, ret = _gae(ep)
        S.extend(ep.states); A.extend(ep.actions); LP.extend(ep.log_probs)
        V.extend(ep.values);  R.extend(ret.tolist()); ADV.extend(adv.tolist()); M.extend(ep.masks)
    if not S: raise RuntimeError("No rollout samples.")
    mk = lambda lst, dt: torch.tensor(np.array(lst), dtype=dt)
    st=mk(S,torch.float32); at=mk(A,torch.long); lpt=mk(LP,torch.float32)
    vt=mk(V,torch.float32); rt=mk(R,torch.float32); mt=mk(M,torch.float32)
    advt = mk(ADV, torch.float32)
    advt = (advt - advt.mean()) / (advt.std() + 1e-8)  # global normalisation
    return st, at, lpt, vt, rt, advt, mt

def collect_rollouts(model, frozen_model, num_games, round_idx, league_pool):
    """
    Collect PPO episodes on fixed training maps.

    Map seed:  _TRAIN_MAP_SEEDS[gi % N_TRAIN_MAPS]  — same 100 maps every round
    Opp seed:  derived from map_seed XOR round/gi   — different opponents each round
    """
    model.eval()
    if frozen_model is not None: frozen_model.eval()
    episodes = []

    for gi in range(num_games):
        map_seed = _TRAIN_MAP_SEEDS[gi % N_TRAIN_MAPS]
        opp_seed = (map_seed + round_idx * 999_983 + gi * 1_000_003) & 0x7FFFFFFF

        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs  = env.reset()
        opps = build_train_opponents(cid, opp_seed, frozen_model, league_pool, round_idx)

        ep = Episode(); done = False; step = 0; trunc_alive = False

        while not done:
            if cid>=len(obs["players"]) or int(obs["players"][cid][2])!=1: break
            state  = encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step).unsqueeze(0).to(DEVICE)
            pos    = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl     = int(obs["players"][cid][3])
            lm     = _legal_mask(obs["map"], obs["bombs"], pos, bl)
            with torch.no_grad():
                action, lp, _, val = _sample_action(model, state, lm, sample=True, temperature=1.0)

            acts = [0,0,0,0]; acts[cid] = action
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            prev_obs = obs
            obs, terminated, truncated = env.step(acts)
            my_died  = int(obs["players"][cid][2]) == 0
            reward   = compute_shaped_reward(prev_obs, obs, cid, action, terminated, truncated)
            genuine_done = bool(my_died or terminated)

            ep.states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
            ep.actions.append(action);    ep.rewards.append(float(reward))
            ep.dones.append(genuine_done); ep.log_probs.append(lp)
            ep.values.append(float(val)); ep.masks.append(lm.astype(np.float32))

            trunc_alive = bool(truncated and not terminated and not my_died)
            done = bool(terminated or truncated or my_died); step += 1

        if trunc_alive and ep.states:
            try:
                ls = encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): _, lv = _fwd(model, ls)
                ep.last_val = float(lv.item())
            except Exception: pass

        if ep.states: episodes.append(ep)
        if (gi+1) % 50 == 0:
            total = sum(len(e.states) for e in episodes)
            print(f"Rollout {gi+1}/{num_games} | eps={len(episodes)} | steps={total}", flush=True)

    return episodes

# ===========================================================================
# PPO update — persistent optimizer passed in from main()
# ===========================================================================
def ppo_update(model, episodes, optimizer, entropy_coef):
    if not episodes: return model
    states, actions, old_lps, _, returns, advantages, masks = _flatten(episodes)
    N = states.shape[0]
    model.train()

    for epoch in range(1, PPO_EPOCHS+1):
        idxs = np.random.permutation(N)
        t_pol = t_val = t_ent = t_tot = n_b = 0.0
        for start in range(0, N, PPO_BATCH_SIZE):
            bi  = idxs[start:start+PPO_BATCH_SIZE]
            if len(bi)==0: continue
            bs  = states[bi].to(DEVICE); ba = actions[bi].to(DEVICE)
            blp = old_lps[bi].to(DEVICE); brt = returns[bi].to(DEVICE)
            bad = advantages[bi].to(DEVICE); bm = masks[bi].to(DEVICE)
            logits, values = _fwd(model, bs)
            ml = logits.clone(); ml[bm<=0] = -1e9
            dist    = Categorical(logits=ml)
            new_lp  = dist.log_prob(ba)
            entropy = dist.entropy().mean()
            ratio   = torch.exp(new_lp - blp)
            clipped = torch.clamp(ratio, 1-PPO_CLIP_EPS, 1+PPO_CLIP_EPS)
            pol_loss= -torch.mean(torch.min(ratio*bad, clipped*bad))
            val_loss= torch.mean((values - brt)**2)
            loss    = pol_loss + PPO_VALUE_COEF*val_loss - entropy_coef*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
            optimizer.step()
            t_pol+=pol_loss.item(); t_val+=val_loss.item()
            t_ent+=entropy.item();  t_tot+=loss.item(); n_b+=1
        nb = max(1, n_b)
        print(f"  PPO {epoch}/{PPO_EPOCHS} | loss={t_tot/nb:.4f} pol={t_pol/nb:.4f} "
              f"val={t_val/nb:.4f} ent={t_ent/nb:.4f}", flush=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# ===========================================================================
# Evaluation — fixed held-out maps, never used in training
# ===========================================================================
def evaluate(model, num_games=20, return_wins=False):
    """
    Eval on _EVAL_MAP_SEEDS — completely separate from training maps.
    Uses stochastic sampling + legal mask (same conditions as training).
    """
    model.eval()
    wins = draws = losses = total_kills = total_steps = 0

    for gi in range(num_games):
        map_seed = _EVAL_MAP_SEEDS[gi % N_EVAL_MAPS]
        opp_seed = map_seed + gi * 9_999_991
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs  = env.reset()
        opps = build_eval_opponents(cid, opp_seed)
        kills = 0; done = False; step = 0

        while not done:
            if int(obs["players"][cid][2]) != 1: break
            state = encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step).unsqueeze(0).to(DEVICE)
            pos   = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl    = int(obs["players"][cid][3])
            lm    = _legal_mask(obs["map"], obs["bombs"], pos, bl)
            with torch.no_grad():
                action, _, _, _ = _sample_action(model, state, lm, sample=True, temperature=0.8)
            prev_e = sum(int(obs["players"][i][2]) for i in range(4) if i!=cid)
            acts = [0,0,0,0]; acts[cid] = action
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            kills += max(0, prev_e - sum(int(obs["players"][i][2]) for i in range(4) if i!=cid))
            done = bool(terminated or truncated); step += 1

        alive = [int(p[2]) for p in obs["players"]]
        if alive[cid]==1 and sum(alive)==1: wins+=1
        elif alive[cid]==1: draws+=1
        else: losses+=1
        total_kills+=kills; total_steps+=step

    ng = max(1, num_games)
    print(f"Eval ({num_games}g) | W={wins} D={draws} L={losses} | "
          f"kills={total_kills/ng:.2f} steps={total_steps/ng:.0f}", flush=True)
    return wins if return_wins else None

def evaluate_on_train(model, num_games=20, return_wins=False):
    """
    Eval on training maps — not a true measure of generalisation, but useful for fast feedback during development.
    Uses deterministic argmax + legal mask (same conditions as training).
    """
    model.eval()
    wins = draws = losses = total_kills = total_steps = 0

    for gi in range(num_games):
        map_seed = _TRAIN_MAP_SEEDS[gi % N_TRAIN_MAPS]
        opp_seed = map_seed + gi * 9_999_991
        cid  = gi % 4
        env  = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs  = env.reset()
        opps = build_train_opponents(cid, opp_seed, None, None, 1)  # all strong for max challenge
        kills = 0; done = False; step = 0

        while not done:
            if int(obs["players"][cid][2]) != 1: break
            state = encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step).unsqueeze(0).to(DEVICE)
            pos   = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl    = int(obs["players"][cid][3])
            lm    = _legal_mask(obs["map"], obs["bombs"], pos, bl)
            with torch.no_grad():
                action, _, _, _ = _sample_action(model, state, lm, sample=False)
            prev_e = sum(int(obs["players"][i][2]) for i in range(4) if i!=cid)
            acts = [0,0,0,0]; acts[cid] = action
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            kills += max(0, prev_e - sum(int(obs["players"][i][2]) for i in range(4) if i!=cid))
            done = bool(terminated or truncated); step += 1

        alive = [int(p[2]) for p in obs["players"]]
        if alive[cid]==1 and sum(alive)==1: wins+=1
        elif alive[cid]==1: draws+=1
        else: losses+=1
        total_kills+=kills; total_steps+=step

    ng = max(1, num_games)
    print(f"Train Eval ({num_games}g) | W={wins} D={draws} L={losses} | "
          f"kills={total_kills/ng:.2f} steps={total_steps/ng:.0f}", flush=True)
    return wins if return_wins else None


# ===========================================================================
# Main — pure self-play from scratch
# ===========================================================================
def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Train maps: {N_TRAIN_MAPS} fixed seeds | Eval maps: {N_EVAL_MAPS} held-out seeds", flush=True)

    model = BomberNet(INPUT_CHANNELS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BomberNet: {n_params:,} parameters", flush=True)
    
    current_dir     = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, "model_bomber.pth")

    # Resume if checkpoint exists
    start_round = 0
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
        print("Starting from random initialization.", flush=True)
    evaluate_on_train(model, num_games=20)

    # Persistent optimizer — momentum carries across rounds
    optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)

    league   = LeaguePool(max_size=LEAGUE_POOL_SIZE)
    league.add(model)

    ent_coef  = float(PPO_ENTROPY_COEF)
    best_wins = -1

    print(f"\n=== Pure self-play PPO — {RL_ROUNDS} rounds ===", flush=True)

    for round_idx in range(start_round, RL_ROUNDS):
        phase = 0 if round_idx<10 else 1 if round_idx<30 else 2 if round_idx<60 else 3
        print(f"\n--- Round {round_idx+1}/{RL_ROUNDS} | phase={phase} | ent={ent_coef:.4f} ---", flush=True)

        # Freeze current model for opponents
        frozen = copy.deepcopy(model).cpu().eval()

        # Collect rollouts on fixed training maps
        rollouts = collect_rollouts(model, frozen, ROLLOUT_GAMES_PER_ROUND, round_idx, league)
        total_steps = sum(len(e.states) for e in rollouts)
        print(f"  collected {len(rollouts)} eps ({total_steps} steps)", flush=True)

        # PPO update
        model = ppo_update(model, rollouts, optimizer, ent_coef)
        league.add(model)

        ent_coef = max(PPO_ENTROPY_MIN, ent_coef * PPO_ENTROPY_DECAY)

        # Evaluate on held-out maps
        wins = evaluate(model, num_games=20, return_wins=True)
        evaluate_on_train(model, num_games=20)

        if wins > best_wins:
            best_wins = wins
            torch.save(model.state_dict(), BEST_PPO_PATH)
            print(f"  ★ New best: {wins}/20 → {BEST_PPO_PATH}", flush=True)

    print("\n=== Final evaluation (50 games) ===", flush=True)
    evaluate(model, num_games=50)

    if os.path.exists(BEST_PPO_PATH):
        print("\n=== Best checkpoint evaluation (50 games) ===", flush=True)
        best = BomberNet(INPUT_CHANNELS).to(DEVICE)
        best.load_state_dict(torch.load(BEST_PPO_PATH, map_location=DEVICE))
        evaluate(best, num_games=50)
        print(f"Best model: {BEST_PPO_PATH}", flush=True)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
