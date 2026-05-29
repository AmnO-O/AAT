"""
agent.py – Inference agent using the trained BomberNet from fat_best_reward_v4.py

Usage:
  1. Ensure model_bc.pth (or model_bc_best.pth) is present.
  2. Import PolicyAgent or run it directly.

The agent expects a dictionary observation with keys 'map', 'players', 'bombs'.
"""

import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
BOARD_SIZE = 13
INPUT_CHANNELS = 27
NUM_ACTIONS = 6
MAX_STEPS = 500
EXPLOSION_TIME_HORIZON = 8.0

# Spatial / scalar channel split (same as pipeline)
SPATIAL_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 21, 24, 25, 26]
SCALAR_CHANNELS = [14, 17, 18, 19, 20, 22, 23]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MOVES = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------
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
    grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
) -> np.ndarray:
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

# ---------------------------------------------------------------------------
# Danger planes (same as training)
# ---------------------------------------------------------------------------
def explosion_time_plane(
    grid, players, bombs, horizon=EXPLOSION_TIME_HORIZON,
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

def danger_plane(grid, players, bombs, timer_threshold=1) -> np.ndarray:
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger
    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / EXPLOSION_TIME_HORIZON
    danger[plane <= threshold] = 1.0
    return danger

def immediate_danger_plane(grid, players, bombs):
    return danger_plane(grid, players, bombs, timer_threshold=1)

def chain_danger_plane(grid, players, bombs, chain_horizon=3) -> np.ndarray:
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

def future_danger_plane(grid, players, bombs, horizon=EXPLOSION_TIME_HORIZON) -> np.ndarray:
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

def tile_earliest_explosion_times(grid, players, bombs) -> np.ndarray:
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

def bomb_pressure_plane(grid, players, bombs, my_id) -> np.ndarray:
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

def future_bomb_pressure_plane(grid, players, bombs, my_id) -> np.ndarray:
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

def bottleneck_risk_plane(grid, players, bombs, my_id) -> np.ndarray:
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

# ---------------------------------------------------------------------------
# Escape / BFS utilities
# ---------------------------------------------------------------------------
def escape_margin_from_position(
    grid, players, bombs, start: Tuple[int, int], max_depth=6,
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

def time_safe_escape_score(grid, players, bombs, my_id) -> float:
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return 0.0
    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    margin = escape_margin_from_position(grid, players, bombs, my_pos, max_depth=6)
    return float(np.clip(margin / 6.0, 0.0, 1.0)) if margin > 0 else 0.0

def bfs_distance_to_targets(
    grid, start, targets: set, bombs, max_depth=64,
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

def bfs_reachable_count(grid, start, bombs, max_depth=3) -> int:
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

def norm_dist(d: Optional[int], cap=24.0) -> float:
    return 1.0 if d is None else float(min(d, cap)) / cap

def normalize_scalar(x: float, denom: float) -> float:
    return float(np.clip(x / denom, 0.0, 1.0)) if denom > 0 else 0.0

# ---------------------------------------------------------------------------
# Bomb safety (with hypothetical bomb)
# ---------------------------------------------------------------------------
def _add_hypothetical_bomb(bombs, pos, owner, timer=7) -> np.ndarray:
    new_row = np.array([[pos[0], pos[1], timer, owner]], dtype=np.int8)
    if bombs is not None and len(bombs) > 0:
        return np.concatenate([bombs, new_row], axis=0)
    return new_row

def should_place_bomb_here(grid, players, bombs, my_id, pos) -> bool:
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return False
    if not passable(grid, pos[0], pos[1]):
        return False
    my_radius = 1 + int(players[my_id][4])
    hyp_bombs = _add_hypothetical_bomb(bombs, pos, my_id)
    blast = blast_tiles(grid, pos[0], pos[1], my_radius)
    blocked = bomb_positions_set(hyp_bombs)
    for a in (1, 2, 3, 4):
        nr, nc = next_pos(pos, a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked or (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6) > 0:
            return True
    return False

def safe_to_bomb_plane(grid, players, bombs, my_id) -> np.ndarray:
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
    hit_boxes = any(int(grid[x, y]) == 2 for x, y in blast)
    hit_enemy = any((x, y) in enemy_positions for x, y in blast)
    if not (hit_boxes or hit_enemy):
        return plane
    hyp_bombs = _add_hypothetical_bomb(bombs, (my_r, my_c), my_id)
    blocked_hyp = bomb_positions_set(hyp_bombs)
    for a in (1, 2, 3, 4):
        nr, nc = next_pos((my_r, my_c), a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked_hyp or (nr, nc) in blast:
            continue
        if escape_margin_from_position(grid, players, hyp_bombs, (nr, nc), max_depth=6) > 0:
            plane[my_r, my_c] = 1.0
            break
    return plane

# ---------------------------------------------------------------------------
# Observation encoding – must exactly match training
# ---------------------------------------------------------------------------
def encode_obs(grid, players, bombs, my_id, step) -> torch.Tensor:
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

    # ch 14 — bombs_left (scalar broadcast)
    state[14].fill(normalize_scalar(bombs_left, 5.0))

    # ch 15-16 — bomb timer / radius heatmaps
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

    # Scalar features (ch 17-20, 22-23)
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

# ---------------------------------------------------------------------------
# Legal / shielded action masks
# ---------------------------------------------------------------------------
def legal_actions(grid, bombs, my_pos, bombs_left) -> List[int]:
    moves = [0]
    blocked = bomb_positions_set(bombs)
    for a in (1, 2, 3, 4):
        nr, nc = next_pos(my_pos, a)
        if passable(grid, nr, nc) and (nr, nc) not in blocked:
            moves.append(a)
    if bombs_left > 0 and my_pos not in blocked:
        moves.append(5)
    return moves

def _legal_action_mask(grid, bombs, my_pos, bombs_left) -> np.ndarray:
    mask = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    for a in legal_actions(grid, bombs, my_pos, bombs_left):
        mask[int(a)] = 1.0
    if mask.sum() <= 0:
        mask[0] = 1.0
    return mask

def _shielded_legal_mask(grid, players, bombs, my_id, legal_mask) -> np.ndarray:
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
                mask[a] = 0.0
                continue
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

# ---------------------------------------------------------------------------
# Model definition – copied exactly from the training pipeline
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)

class BomberNet(nn.Module):
    _SPATIAL = SPATIAL_CHANNELS
    _SCALAR  = SCALAR_CHANNELS
    _POOL    = 4

    def __init__(self, input_channels=INPUT_CHANNELS, num_actions=NUM_ACTIONS, width=64):
        super().__init__()
        n_sp = len(self._SPATIAL)
        n_sc = len(self._SCALAR)
        feat_dim = width * (self._POOL ** 2) + n_sc

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
        self.register_buffer("_sp_idx", torch.tensor(self._SPATIAL, dtype=torch.long))
        self.register_buffer("_sc_idx", torch.tensor(self._SCALAR, dtype=torch.long))

    def forward(self, x):
        sp = x[:, self._sp_idx]               # (B, 20, 13, 13)
        sc = x[:, self._sc_idx, 0, 0]         # (B, 7)
        feat = self.stem(sp)
        feat = self.blocks(feat)
        feat = self.pool(feat).flatten(1)      # (B, 1024)
        combined = torch.cat([feat, sc], dim=1)
        logits = self.policy_head(combined)
        value  = self.value_head(combined).squeeze(-1)
        return logits, value

# ---------------------------------------------------------------------------
# Inference agent
# ---------------------------------------------------------------------------
class PolicyAgent:
    """
    Agent that uses the trained BomberNet model.

    Args:
        agent_id: player index (0-3) for this agent.
        model_path: path to the saved .pth file (e.g. 'model_bc_best.pth').
        deterministic: if True, always pick the argmax; if False, sample.
    """
    def __init__(self, agent_id: int, model_path: str = "model_bc.pth",
                 deterministic: bool = False):
        self.agent_id = int(agent_id)
        self.deterministic = deterministic
        self.model = BomberNet(INPUT_CHANNELS, NUM_ACTIONS).to(DEVICE)
        self.model.eval()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_dir, model_path)

        if os.path.exists(full_model_path):
            state_dict = torch.load(full_model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            print(f"Successfully loaded model from {full_model_path}")
        else:
            print(f"WARNING: model file {full_model_path} not found, using random weights.")

        self._step = 0

    def reset(self):
        self._step = 0

    def act(self, obs: Dict) -> int:
        # Check if agent is dead
        if self.agent_id >= len(obs["players"]) or int(obs["players"][self.agent_id][2]) != 1:
            self._step += 1
            return 0

        step = self._step
        self._step += 1

        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        # Encode observation
        state = encode_obs(grid, players, bombs, self.agent_id, step).unsqueeze(0).to(DEVICE)

        # Legal / shielded mask
        my_pos     = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
        bombs_left = int(players[self.agent_id][3])
        lm = _legal_action_mask(grid, bombs, my_pos, bombs_left)
        shield = _shielded_legal_mask(grid, players, bombs, self.agent_id, lm)

        # Model inference
        with torch.no_grad():
            logits, _ = self.model(state)
            mask_t = torch.tensor(shield, dtype=torch.bool, device=logits.device).unsqueeze(0)
            masked = logits.clone()
            masked[~mask_t] = -1e9

            if self.deterministic:
                action = torch.argmax(masked, dim=-1)
            else:
                dist = Categorical(logits=masked)
                action = dist.sample()

        return int(action.item())

