
import os
import json
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# -----------------------------------------------------------------------------
# Optional local contest import
# -----------------------------------------------------------------------------
try:
    from engine.game import BomberEnv  # type: ignore
except Exception:  # pragma: no cover
    BomberEnv = None  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BOARD_SIZE = 13
MAX_STEPS = 500
NUM_ACTIONS = 6
INPUT_CHANNELS = 24

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INITIAL_GAMES = 800
DAGGER_ROUNDS = 2
DAGGER_GAMES_PER_ROUND = 200

TRAIN_SPLIT_MOD = 10
CHUNK_SIZE = 2048
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
PATIENCE = 5
GRAD_CLIP_NORM = 1.0

TRAIN_DIR = "bc_train_chunks"
VAL_DIR = "bc_val_chunks"
MODEL_PATH = "model_bc.pth"
BEST_MODEL_PATH = "model_bc_best.pth"
MANIFEST_NAME = "manifest.json"

PASSABLE = {0, 3, 4}
STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = range(6)
ACTION_DELTAS = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TARGET_ITEMS = {3, 4}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def transform_coord(r: int, c: int, agent_id: int) -> Tuple[int, int]:
    agent_id = int(agent_id)
    if agent_id == 0:
        return r, c
    if agent_id == 1:
        return BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c
    if agent_id == 2:
        return r, BOARD_SIZE - 1 - c
    if agent_id == 3:
        return BOARD_SIZE - 1 - r, c
    return r, c


def transform_action(action: int, agent_id: int) -> int:
    """
    Map action between world and canonical space.

    The transform is self-inverse for the four seat symmetries we use:
    - id 0: identity
    - id 1: rotate 180 degrees
    - id 2: mirror horizontally
    - id 3: mirror vertically
    """
    a = int(action)
    agent_id = int(agent_id)
    if a not in range(6):
        return STOP
    if agent_id == 0:
        return a
    if agent_id == 1:
        return {LEFT: RIGHT, RIGHT: LEFT, UP: DOWN, DOWN: UP}.get(a, a)
    if agent_id == 2:
        return {LEFT: RIGHT, RIGHT: LEFT}.get(a, a)
    if agent_id == 3:
        return {UP: DOWN, DOWN: UP}.get(a, a)
    return a


def canonicalize_grid(grid: np.ndarray, agent_id: int) -> np.ndarray:
    if agent_id == 0:
        return np.array(grid, copy=True)
    if agent_id == 1:
        return np.flipud(np.fliplr(grid)).copy()
    if agent_id == 2:
        return np.fliplr(grid).copy()
    if agent_id == 3:
        return np.flipud(grid).copy()
    return np.array(grid, copy=True)


def canonicalize_players(players: np.ndarray, agent_id: int) -> np.ndarray:
    out = np.array(players, copy=True)
    for i in range(len(out)):
        r, c = int(out[i][0]), int(out[i][1])
        nr, nc = transform_coord(r, c, agent_id)
        out[i][0] = nr
        out[i][1] = nc
    return out


def canonicalize_bombs(bombs: np.ndarray, agent_id: int) -> np.ndarray:
    if bombs is None or len(bombs) == 0:
        return np.zeros((0, 4), dtype=np.int8)
    out = np.array(bombs, copy=True)
    for i in range(len(out)):
        r, c = int(out[i][0]), int(out[i][1])
        nr, nc = transform_coord(r, c, agent_id)
        out[i][0] = nr
        out[i][1] = nc
    return out


def canonicalize_obs(obs: Dict, agent_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = canonicalize_grid(np.asarray(obs["map"]), agent_id)
    players = canonicalize_players(np.asarray(obs["players"]), agent_id)
    bombs = canonicalize_bombs(np.asarray(obs["bombs"]), agent_id)
    return grid, players, bombs


def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in PASSABLE


def move_from(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    dr, dc = ACTION_DELTAS.get(int(action), (0, 0))
    return pos[0] + dr, pos[1] + dc


def bomb_positions_set(bombs: np.ndarray) -> set:
    if bombs is None or len(bombs) == 0:
        return set()
    return {(int(b[0]), int(b[1])) for b in bombs}


def my_info(players: np.ndarray, agent_id: int) -> Tuple[bool, Tuple[int, int], int, int]:
    if agent_id >= len(players):
        return False, (0, 0), 0, 1
    alive = int(players[agent_id][2]) == 1
    pos = (int(players[agent_id][0]), int(players[agent_id][1]))
    bombs_left = int(players[agent_id][3])
    bomb_radius = 1 + int(players[agent_id][4])
    return alive, pos, bombs_left, bomb_radius


# -----------------------------------------------------------------------------
# Bomb simulation
# -----------------------------------------------------------------------------
def bomb_radius_for(bombs: np.ndarray, players: np.ndarray, bomb_idx: int) -> int:
    owner = int(bombs[bomb_idx][3]) if bombs is not None and len(bombs) > bomb_idx else -1
    if 0 <= owner < len(players):
        # Contest observations do not expose per-bomb radius, so we approximate
        # using the owner's current radius bonus.
        return max(1, 1 + int(players[owner][4]))
    return 1


def blast_tiles(grid: np.ndarray, center: Tuple[int, int], radius: int) -> set:
    r0, c0 = center
    tiles = {(r0, c0)}
    for dr, dc in DIRECTIONS:
        for d in range(1, radius + 1):
            r, c = r0 + dr * d, c0 + dc * d
            if not in_bounds(r, c):
                break
            cell = int(grid[r, c])
            if cell == 1:
                break
            tiles.add((r, c))
            if cell == 2:
                break
    return tiles


def bomb_explosion_times(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    """
    Exact-ish chain reaction closure under the contest rules:
    bombs in the blast path of an earlier explosion detonate immediately
    in the same step, so we relax explosion times until a fixpoint.
    """
    n = 0 if bombs is None else len(bombs)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    base = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    times = base.copy()

    blasts: List[set] = []
    for i in range(n):
        radius = bomb_radius_for(bombs, players, i)
        blasts.append(blast_tiles(grid, (int(bombs[i][0]), int(bombs[i][1])), radius))

    changed = True
    while changed:
        changed = False
        for i in range(n):
            ti = int(times[i])
            for j in range(n):
                if i == j:
                    continue
                if (int(bombs[j][0]), int(bombs[j][1])) in blasts[i] and times[j] > ti:
                    times[j] = ti
                    changed = True
    return times


def danger_map(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, horizon: int = 1) -> np.ndarray:
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger
    times = bomb_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        if int(times[i]) <= int(horizon):
            radius = bomb_radius_for(bombs, players, i)
            for r, c in blast_tiles(grid, (int(bombs[i][0]), int(bombs[i][1])), radius):
                danger[r, c] = 1.0
    return danger


# -----------------------------------------------------------------------------
# Pathfinding / tactical metrics
# -----------------------------------------------------------------------------
def bfs_distance_map(grid: np.ndarray, start: Tuple[int, int], bombs: np.ndarray, max_depth: int = 64) -> np.ndarray:
    dist = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int16)
    blocked = bomb_positions_set(bombs)
    if start in blocked:
        blocked = set(blocked)
        blocked.discard(start)

    q = deque([(start[0], start[1])])
    dist[start[0], start[1]] = 0

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        if d >= max_depth:
            continue
        nd = d + 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not passable(grid, nr, nc):
                continue
            if (nr, nc) in blocked:
                continue
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def local_degree_map(grid: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    blocked = bomb_positions_set(bombs)
    deg = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not passable(grid, r, c) or (r, c) in blocked:
                continue
            cnt = 0
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if passable(grid, nr, nc) and (nr, nc) not in blocked:
                    cnt += 1
            deg[r, c] = cnt / 4.0
    return deg


def local_box_density(grid: np.ndarray, radius: int = 2) -> np.ndarray:
    boxes = (grid == 2).astype(np.float32)
    out = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        r0 = max(0, r - radius)
        r1 = min(BOARD_SIZE, r + radius + 1)
        for c in range(BOARD_SIZE):
            c0 = max(0, c - radius)
            c1 = min(BOARD_SIZE, c + radius + 1)
            out[r, c] = float(boxes[r0:r1, c0:c1].sum()) / float((r1 - r0) * (c1 - c0))
    return out


def enemy_trap_potential(grid: np.ndarray, players: np.ndarray, agent_id: int, bombs: np.ndarray) -> np.ndarray:
    out = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    blocked = bomb_positions_set(bombs)
    for pid in range(len(players)):
        if pid == agent_id or int(players[pid][2]) != 1:
            continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if not in_bounds(r, c):
            continue
        if (r, c) in blocked or not passable(grid, r, c):
            continue
        free_n = 0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if passable(grid, nr, nc) and (nr, nc) not in blocked:
                free_n += 1
        trap = 1.0 - float(free_n) / 4.0
        out[r, c] = max(out[r, c], trap)
    return out


def norm_distance_map(dist: np.ndarray, cap: int = 24) -> np.ndarray:
    out = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    mask = dist >= 0
    out[mask] = np.minimum(dist[mask], cap).astype(np.float32) / float(cap)
    return out


def build_safe_reachability(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    danger = danger_map(grid, players, bombs, horizon=1)
    dist = bfs_distance_map(grid, start, bombs, max_depth=16)
    reachable = ((dist >= 0) & (danger == 0.0)).astype(np.float32)
    return reachable


def can_escape_after_bomb(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bomb_radius: int) -> bool:
    """
    Conservative escape check: assume the current tile becomes dangerous after
    placing the bomb, then look for any safe reachable tile within 6 steps.
    """
    future = danger_map(grid, players, bombs, horizon=1)
    bomb_tiles = blast_tiles(grid, pos, bomb_radius)
    # Treat the hypothetical bomb blast as unsafe.
    unsafe = set(bomb_tiles)
    blocked = bomb_positions_set(bombs)
    q = deque([(pos[0], pos[1], 0)])
    seen = {pos}
    while q:
        r, c, d = q.popleft()
        if d > 0 and (future[r, c] == 0.0) and ((r, c) not in unsafe):
            return True
        if d >= 6:
            continue
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if not passable(grid, nr, nc):
                continue
            if (nr, nc) in blocked:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, d + 1))
    return False


def count_boxes_in_blast(grid: np.ndarray, pos: Tuple[int, int], radius: int) -> int:
    return sum(1 for r, c in blast_tiles(grid, pos, radius) if int(grid[r, c]) == 2)


def can_hit_enemy_with_bomb(grid: np.ndarray, players: np.ndarray, agent_id: int, pos: Tuple[int, int], radius: int) -> bool:
    r0, c0 = pos
    for pid in range(len(players)):
        if pid == agent_id or int(players[pid][2]) != 1:
            continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if r == r0:
            step = 1 if c > c0 else -1
            clear = True
            for cc in range(c0 + step, c, step):
                if int(grid[r0, cc]) in (1, 2):
                    clear = False
                    break
            if clear and abs(c - c0) <= radius:
                return True
        if c == c0:
            step = 1 if r > r0 else -1
            clear = True
            for rr in range(r0 + step, r, step):
                if int(grid[rr, c0]) in (1, 2):
                    clear = False
                    break
            if clear and abs(r - r0) <= radius:
                return True
    return False


# -----------------------------------------------------------------------------
# Observation encoding
# -----------------------------------------------------------------------------
def encode_features(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, agent_id: int, step: int) -> np.ndarray:
    """
    24 planes on a canonical board:
      0 walls
      1 boxes
      2 grass
      3 radius items
      4 capacity items
      5 my position
      6 all enemies
      7 nearest enemy
      8 all bombs
      9 my bombs
      10 enemy bombs
      11 bomb timer heat
      12 immediate danger
      13 future danger
      14 reachable safe area
      15 safe-to-bomb plane
      16 dead-end / low-degree plane
      17 item distance
      18 enemy distance
      19 local box density
      20 enemy trap potential
      21 bombs left norm
      22 bomb radius norm
      23 phase ratio
    """
    canon_grid, canon_players, canon_bombs = canonicalize_obs({"map": grid, "players": players, "bombs": bombs}, agent_id)
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Static map
    state[0] = (canon_grid == 1).astype(np.float32)
    state[1] = (canon_grid == 2).astype(np.float32)
    state[2] = (canon_grid == 0).astype(np.float32)
    state[3] = (canon_grid == 3).astype(np.float32)
    state[4] = (canon_grid == 4).astype(np.float32)

    alive, my_pos, bombs_left, bomb_radius = my_info(canon_players, agent_id)

    # Players
    if alive and in_bounds(*my_pos):
        state[5, my_pos[0], my_pos[1]] = 1.0

    enemy_positions = []
    for pid in range(len(canon_players)):
        if pid == agent_id or int(canon_players[pid][2]) != 1:
            continue
        r, c = int(canon_players[pid][0]), int(canon_players[pid][1])
        if in_bounds(r, c):
            state[6, r, c] = 1.0
            enemy_positions.append((r, c))

    if alive and enemy_positions:
        # nearest enemy in canonical space
        dist_map = bfs_distance_map(canon_grid, my_pos, canon_bombs, max_depth=64)
        best_enemy = None
        best_dist = 10**9
        for ep in enemy_positions:
            d = int(dist_map[ep[0], ep[1]])
            if d >= 0 and d < best_dist:
                best_dist = d
                best_enemy = ep
        if best_enemy is not None:
            state[7, best_enemy[0], best_enemy[1]] = 1.0

    # Bombs
    if canon_bombs is not None and len(canon_bombs) > 0:
        times = bomb_explosion_times(canon_grid, canon_players, canon_bombs)
        for i in range(len(canon_bombs)):
            r, c = int(canon_bombs[i][0]), int(canon_bombs[i][1])
            if not in_bounds(r, c):
                continue
            owner = int(canon_bombs[i][3]) if canon_bombs.shape[1] > 3 else -1
            state[8, r, c] = 1.0
            if owner == agent_id:
                state[9, r, c] = 1.0
            elif 0 <= owner < len(canon_players):
                state[10, r, c] = 1.0
            state[11, r, c] = 1.0 / float(max(1, int(canon_bombs[i][2]) + 1))
    else:
        times = np.zeros((0,), dtype=np.int32)

    immediate = danger_map(canon_grid, canon_players, canon_bombs, horizon=1)
    future = danger_map(canon_grid, canon_players, canon_bombs, horizon=3)
    state[12] = immediate
    state[13] = future

    if alive:
        safe_reach = build_safe_reachability(canon_grid, canon_players, canon_bombs, my_pos)
        state[14] = safe_reach

        can_bomb = 1.0 if (bombs_left > 0 and not (canon_bombs is not None and (my_pos in bomb_positions_set(canon_bombs))) and can_escape_after_bomb(canon_grid, canon_players, canon_bombs, my_pos, bomb_radius)) else 0.0
        state[15, my_pos[0], my_pos[1]] = can_bomb

        deg = local_degree_map(canon_grid, canon_bombs)
        state[16] = (deg <= 0.25).astype(np.float32)

        dist_item = bfs_distance_map(canon_grid, my_pos, canon_bombs, max_depth=64)
        item_targets = [(int(r), int(c)) for r, c in np.argwhere((canon_grid == 3) | (canon_grid == 4))]
        enemy_targets = enemy_positions

        if item_targets:
            best = min((int(dist_item[r, c]) for r, c in item_targets if int(dist_item[r, c]) >= 0), default=-1)
        else:
            best = -1
        state[17] = norm_distance_map(np.full((BOARD_SIZE, BOARD_SIZE), best, dtype=np.int16), cap=24)

        if enemy_targets:
            best_e = min((int(dist_item[r, c]) for r, c in enemy_targets if int(dist_item[r, c]) >= 0), default=-1)
        else:
            best_e = -1
        state[18] = norm_distance_map(np.full((BOARD_SIZE, BOARD_SIZE), best_e, dtype=np.int16), cap=24)

        state[19] = local_box_density(canon_grid, radius=2)
        state[20] = enemy_trap_potential(canon_grid, canon_players, agent_id, canon_bombs)
    else:
        state[14] = 0.0
        state[16] = 0.0
        state[17] = 1.0
        state[18] = 1.0

    state[21] = float(bombs_left) / 5.0
    state[22] = float(bomb_radius) / 6.0
    state[23] = float(step) / float(MAX_STEPS)

    return state


# -----------------------------------------------------------------------------
# Heuristic teachers / opponents
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyProfile:
    name: str
    danger_margin: float = 1.0
    box_bonus: float = 1.0
    enemy_bonus: float = 1.0
    bomb_aggression: float = 1.0
    item_bias: float = 1.0


PROFILE_POOL: List[PolicyProfile] = [
    PolicyProfile("balanced", danger_margin=1.0, box_bonus=1.2, enemy_bonus=1.1, bomb_aggression=1.0, item_bias=1.0),
    PolicyProfile("cautious", danger_margin=1.4, box_bonus=0.8, enemy_bonus=0.9, bomb_aggression=0.7, item_bias=1.3),
    PolicyProfile("aggressive", danger_margin=0.8, box_bonus=1.4, enemy_bonus=1.4, bomb_aggression=1.4, item_bias=0.8),
    PolicyProfile("farmer", danger_margin=1.0, box_bonus=1.6, enemy_bonus=0.8, bomb_aggression=1.1, item_bias=0.8),
]


class HeuristicAgent:
    def __init__(self, agent_id: int, profile: PolicyProfile = PROFILE_POOL[0], rng: Optional[random.Random] = None):
        self.agent_id = int(agent_id)
        self.profile = profile
        self.rng = rng or random.Random(0)

    def act(self, obs: Dict) -> int:
        try:
            grid, players, bombs = canonicalize_obs(obs, self.agent_id)
            alive, pos, bombs_left, bomb_radius = my_info(players, self.agent_id)
            if not alive:
                return STOP

            danger_now = danger_map(grid, players, bombs, horizon=1)
            blocked = bomb_positions_set(bombs)

            if danger_now[pos[0], pos[1]] > 0.0:
                escape = self._escape_action(grid, players, bombs, pos)
                if escape is not None:
                    return transform_action(escape, self.agent_id)

            legal = self._legal_actions(grid, bombs, pos, bombs_left)
            scores = {a: -1e9 for a in legal}
            if STOP in scores:
                scores[STOP] = -0.15

            dist_map = bfs_distance_map(grid, pos, bombs, max_depth=64)
            item_targets = [(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))]
            enemy_targets = [(int(players[pid][0]), int(players[pid][1])) for pid in range(len(players)) if pid != self.agent_id and int(players[pid][2]) == 1]

            # Prefer movement that improves access to items or enemies.
            for a in [LEFT, RIGHT, UP, DOWN]:
                if a not in legal:
                    continue
                nxt = move_from(pos, a)
                if danger_now[nxt[0], nxt[1]] > 0.0:
                    scores[a] = -5.0
                    continue
                base = 0.0
                if item_targets:
                    cur = min((int(dist_map[r, c]) for r, c in item_targets if int(dist_map[r, c]) >= 0), default=999)
                    nd_map = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                    nxt_best = min((int(nd_map[r, c]) for r, c in item_targets if int(nd_map[r, c]) >= 0), default=999)
                    if nxt_best < cur:
                        base += self.profile.item_bias * (cur - nxt_best + 1) * 0.12
                if enemy_targets:
                    cur_e = min((int(dist_map[r, c]) for r, c in enemy_targets if int(dist_map[r, c]) >= 0), default=999)
                    nd_map = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                    nxt_e = min((int(nd_map[r, c]) for r, c in enemy_targets if int(nd_map[r, c]) >= 0), default=999)
                    if nxt_e < cur_e:
                        base += self.profile.enemy_bonus * (cur_e - nxt_e + 1) * 0.06
                # Staying near open space helps surviving after bombs.
                deg = 0
                for dr, dc in DIRECTIONS:
                    nr, nc = nxt[0] + dr, nxt[1] + dc
                    if passable(grid, nr, nc) and (nr, nc) not in blocked:
                        deg += 1
                base += 0.03 * deg
                scores[a] = base

            if PLACE_BOMB in legal:
                boxes = count_boxes_in_blast(grid, pos, bomb_radius)
                enemy_hit = can_hit_enemy_with_bomb(grid, players, self.agent_id, pos, bomb_radius)
                escape_ok = can_escape_after_bomb(grid, players, bombs, pos, bomb_radius)
                bomb_score = 0.0
                bomb_score += self.profile.box_bonus * boxes * 0.9
                bomb_score += self.profile.enemy_bonus * (2.5 if enemy_hit else 0.0)
                bomb_score += self.profile.bomb_aggression * (0.2 if escape_ok else -2.0)
                if boxes == 0 and not enemy_hit:
                    bomb_score -= 0.7
                scores[PLACE_BOMB] = bomb_score

            # discourage waiting in open safe positions with useful moves
            if STOP in scores:
                if any(scores[a] > scores[STOP] for a in [LEFT, RIGHT, UP, DOWN] if a in scores):
                    scores[STOP] -= 0.2

            # tie-break deterministically but with a tiny seeded jitter for diversity
            best = max(scores.items(), key=lambda kv: (kv[1], self.rng.random()))[0]
            return transform_action(best, self.agent_id)
        except Exception:
            return STOP

    def _legal_actions(self, grid: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bombs_left: int) -> List[int]:
        legal = [STOP]
        blocked = bomb_positions_set(bombs)
        for a in [LEFT, RIGHT, UP, DOWN]:
            nxt = move_from(pos, a)
            if passable(grid, nxt[0], nxt[1]) and nxt not in blocked:
                legal.append(a)
        if bombs_left > 0 and pos not in blocked:
            legal.append(PLACE_BOMB)
        return legal

    def _escape_action(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, start: Tuple[int, int]) -> Optional[int]:
        blocked = bomb_positions_set(bombs)
        danger = danger_map(grid, players, bombs, horizon=1)
        q = deque([(start[0], start[1], None, 0)])
        seen = {start}
        while q:
            r, c, first, dist = q.popleft()
            if dist > 0 and danger[r, c] == 0.0:
                return first
            if dist >= 6:
                continue
            for a in [LEFT, RIGHT, UP, DOWN]:
                nr, nc = move_from((r, c), a)
                if (nr, nc) in seen:
                    continue
                if not passable(grid, nr, nc):
                    continue
                if (nr, nc) in blocked:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc, a if first is None else first, dist + 1))
        return None


class TeacherPolicy:
    def __init__(self, agent_id: int, profile: Optional[PolicyProfile] = None):
        self.agent_id = int(agent_id)
        self.profile = profile or random.choice(PROFILE_POOL)
        self.bot = HeuristicAgent(agent_id, profile=self.profile, rng=random.Random(SEED + agent_id * 997 + hash(self.profile.name) % 10000))

    def act(self, obs: Dict) -> int:
        return int(self.bot.act(obs))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.04):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = torch.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return torch.relu(x + y)


class BomberNet(nn.Module):
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_actions: int = NUM_ACTIONS, width: int = 48, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
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
        x = self.body(x)
        x = self.pool(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# Chunked dataset
# -----------------------------------------------------------------------------
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
    states_np = np.stack(states, axis=0).astype(np.float16)
    actions_np = np.asarray(actions, dtype=np.int64)
    seeds_np = np.asarray(seeds, dtype=np.int64)
    hist = np.bincount(actions_np, minlength=NUM_ACTIONS).astype(int).tolist()

    filename = f"chunk_{chunk_idx:05d}.npz"
    path = os.path.join(chunk_dir, filename)
    np.savez_compressed(path, states=states_np, actions=actions_np, seeds=seeds_np)
    return {
        "file": filename,
        "count": int(len(actions_np)),
        "action_hist": hist,
        "seed_min": int(seeds_np.min()) if len(seeds_np) else None,
        "seed_max": int(seeds_np.max()) if len(seeds_np) else None,
    }


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
        self.total_len = int(sum(int(c.get("count", 0)) for c in self.chunks))

    def __len__(self) -> int:
        return self.total_len

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (0 if worker is None else worker.id * 10007))
        indices = np.arange(len(self.chunks))
        if self.shuffle_chunks:
            rng.shuffle(indices)
        if worker is not None and worker.num_workers > 1:
            indices = indices[worker.id :: worker.num_workers]

        for idx in indices:
            meta = self.chunks[int(idx)]
            path = os.path.join(self.chunk_dir, meta["file"])
            data = np.load(path)
            states = data["states"]
            actions = data["actions"]
            order = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(order)
            for i in order:
                state = torch.from_numpy(states[int(i)]).float()
                action = int(actions[int(i)])
                if self.augment:
                    state, action = augment_state_and_action(state, action, rng)
                yield state, torch.tensor(action, dtype=torch.long)


# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------
def augment_state_and_action(state: torch.Tensor, action: int, rng: np.random.Generator) -> Tuple[torch.Tensor, int]:
    p = float(rng.random())
    if p < 0.25:
        state = torch.flip(state, dims=[2])  # horizontal
        action = transform_action(action, 2)
    elif p < 0.50:
        state = torch.flip(state, dims=[1])  # vertical
        action = transform_action(action, 3)
    elif p < 0.75:
        state = torch.flip(state, dims=[1, 2])  # 180°
        action = transform_action(transform_action(action, 2), 3)
    # else: no-op
    return state, int(action)


def compute_class_weights(chunk_dir: str) -> torch.Tensor:
    manifest = load_manifest(chunk_dir)
    total = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for chunk in manifest.get("chunks", []):
        total += np.array(chunk.get("action_hist", [0] * NUM_ACTIONS), dtype=np.float64)
    total = np.maximum(total, 1.0)
    w = total.sum() / total
    w = w / w.mean()
    w = np.clip(w, 0.5, 5.0)
    return torch.tensor(w, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def make_env(seed: int):
    if BomberEnv is None:
        raise RuntimeError("BomberEnv is unavailable. Place this script inside the contest kit repo.")
    try:
        return BomberEnv(max_steps=MAX_STEPS, seed=seed)
    except TypeError:
        return BomberEnv(seed=seed)


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env, actions: Sequence[int]):
    out = env.step(list(map(int, actions)))
    if len(out) == 4:
        obs, terminated, truncated, _info = out
    else:
        obs, terminated, truncated = out[:3]
    return obs, bool(terminated), bool(truncated)


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------
def build_opponents(controlled_id: int, game_seed: int) -> Dict[int, HeuristicAgent]:
    rng = random.Random(game_seed)
    ids = [pid for pid in range(4) if pid != controlled_id]
    opponents = {}
    for pid in ids:
        profile = rng.choice(PROFILE_POOL)
        opponents[pid] = HeuristicAgent(pid, profile=profile, rng=random.Random(game_seed + pid * 1337))
    return opponents


def collect_one_game(seed: int, split: str, train_buf, val_buf, train_manifest, val_manifest, teacher_profile: PolicyProfile):
    env = make_env(seed)
    obs = reset_env(env)

    controlled_id = random.Random(seed).randint(0, 3)
    teacher = TeacherPolicy(controlled_id, profile=teacher_profile)
    opponents = build_opponents(controlled_id, seed)

    done = False
    step = 0
    while not done:
        state = encode_features(obs["map"], obs["players"], obs["bombs"], controlled_id, step)
        raw_action = int(teacher.act(obs))
        canonical_action = transform_action(raw_action, controlled_id)

        if split == "train":
            train_buf["states"].append(state.astype(np.float32))
            train_buf["actions"].append(canonical_action)
            train_buf["seeds"].append(seed)
        else:
            val_buf["states"].append(state.astype(np.float32))
            val_buf["actions"].append(canonical_action)
            val_buf["seeds"].append(seed)

        actions = [STOP, STOP, STOP, STOP]
        actions[controlled_id] = raw_action
        for pid, opp in opponents.items():
            actions[pid] = int(opp.act(obs))

        obs, terminated, truncated = step_env(env, actions)
        done = terminated or truncated
        step += 1

        if split == "train" and len(train_buf["states"]) >= CHUNK_SIZE:
            entry = flush_chunk(TRAIN_DIR, train_buf["chunk_idx"], train_buf["states"], train_buf["actions"], train_buf["seeds"])
            if entry:
                train_manifest["chunks"].append(entry)
                save_manifest(TRAIN_DIR, train_manifest)
                train_buf["chunk_idx"] += 1
            train_buf["states"].clear()
            train_buf["actions"].clear()
            train_buf["seeds"].clear()

        if split == "val" and len(val_buf["states"]) >= CHUNK_SIZE:
            entry = flush_chunk(VAL_DIR, val_buf["chunk_idx"], val_buf["states"], val_buf["actions"], val_buf["seeds"])
            if entry:
                val_manifest["chunks"].append(entry)
                save_manifest(VAL_DIR, val_manifest)
                val_buf["chunk_idx"] += 1
            val_buf["states"].clear()
            val_buf["actions"].clear()
            val_buf["seeds"].clear()


def collect_initial_data(num_games: int) -> None:
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    train_manifest = load_manifest(TRAIN_DIR)
    val_manifest = load_manifest(VAL_DIR)

    train_buf = {"states": [], "actions": [], "seeds": [], "chunk_idx": len(train_manifest.get("chunks", []))}
    val_buf = {"states": [], "actions": [], "seeds": [], "chunk_idx": len(val_manifest.get("chunks", []))}

    rng = random.Random(SEED)

    for game_idx in range(num_games):
        seed = SEED * 1000 + game_idx
        split = "val" if (seed % TRAIN_SPLIT_MOD == 0) else "train"
        teacher_profile = rng.choice(PROFILE_POOL)
        collect_one_game(seed, split, train_buf, val_buf, train_manifest, val_manifest, teacher_profile)

        if (game_idx + 1) % 50 == 0:
            train_count = sum(c["count"] for c in train_manifest.get("chunks", [])) + len(train_buf["actions"])
            val_count = sum(c["count"] for c in val_manifest.get("chunks", [])) + len(val_buf["actions"])
            print(f"Collected {game_idx + 1}/{num_games} games | train={train_count} | val={val_count}", flush=True)

    if train_buf["states"]:
        entry = flush_chunk(TRAIN_DIR, train_buf["chunk_idx"], train_buf["states"], train_buf["actions"], train_buf["seeds"])
        if entry:
            train_manifest["chunks"].append(entry)
    if val_buf["states"]:
        entry = flush_chunk(VAL_DIR, val_buf["chunk_idx"], val_buf["states"], val_buf["actions"], val_buf["seeds"])
        if entry:
            val_manifest["chunks"].append(entry)

    save_manifest(TRAIN_DIR, train_manifest)
    save_manifest(VAL_DIR, val_manifest)


def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int) -> int:
    ensure_dir(out_dir)
    model.eval()

    manifest = load_manifest(out_dir)
    buf = {"states": [], "actions": [], "seeds": [], "chunk_idx": len(manifest.get("chunks", []))}
    collected = 0

    def flush_buf():
        nonlocal collected
        if not buf["states"]:
            return
        entry = flush_chunk(out_dir, buf["chunk_idx"], buf["states"], buf["actions"], buf["seeds"])
        if entry:
            manifest["chunks"].append(entry)
            save_manifest(out_dir, manifest)
            buf["chunk_idx"] += 1
            collected += entry["count"]
        buf["states"].clear()
        buf["actions"].clear()
        buf["seeds"].clear()

    rng = random.Random(SEED + 999)

    for game_idx in range(num_games):
        seed = 100000 + SEED * 1000 + game_idx
        controlled_id = rng.randint(0, 3)
        teacher = TeacherPolicy(controlled_id, profile=rng.choice(PROFILE_POOL))
        opponents = build_opponents(controlled_id, seed)

        env = make_env(seed)
        obs = reset_env(env)
        done = False
        step = 0

        while not done:
            state = encode_features(obs["map"], obs["players"], obs["bombs"], controlled_id, step)
            with torch.no_grad():
                inp = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                logits = model(inp)[0]
                student_action = int(torch.argmax(logits).item())
            teacher_action_raw = int(teacher.act(obs))
            teacher_action = transform_action(teacher_action_raw, controlled_id)

            if student_action != teacher_action:
                buf["states"].append(state.astype(np.float32))
                buf["actions"].append(teacher_action)
                buf["seeds"].append(seed)

            actions = [STOP, STOP, STOP, STOP]
            actions[controlled_id] = transform_action(student_action, controlled_id)
            for pid, opp in opponents.items():
                actions[pid] = int(opp.act(obs))

            obs, terminated, truncated = step_env(env, actions)
            done = terminated or truncated
            step += 1

            if len(buf["states"]) >= CHUNK_SIZE:
                flush_buf()

        if (game_idx + 1) % 50 == 0:
            print(f"DAgger {game_idx + 1}/{num_games} games | new samples={collected + len(buf['actions'])}", flush=True)

    flush_buf()
    return collected


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def build_loaders(train_dir: str, val_dir: str):
    train_ds = ChunkedBomberIterableDataset(train_dir, augment=True, shuffle_chunks=True, shuffle_within_chunk=True, seed=SEED)
    val_ds = ChunkedBomberIterableDataset(val_dir, augment=False, shuffle_chunks=False, shuffle_within_chunk=False, seed=SEED)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation samples found in {val_dir}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
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

        total_loss += float(loss.item()) * int(states.size(0))
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == actions).sum().item())
        total_count += int(states.size(0))

        if batch_idx % 50 == 0:
            print(f"    batch {batch_idx}/{max(1, len(loader))}", flush=True)
        
        print(f"    batch {batch_idx}/{max(1, len(loader))} | loss={loss.item():.4f} | acc={int((preds == actions).sum().item()) / max(1, states.size(0)):.4f}", flush=True)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def train_policy_model(train_dir: str, val_dir: str, init_model_path: Optional[str] = None, lr: float = LEARNING_RATE):
    train_loader, val_loader, class_weights = build_loaders(train_dir, val_dir)

    model = BomberNet().to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        try:
            state = torch.load(init_model_path, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"Warning: could not load init model {init_model_path}: {e}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
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


def quick_eval_against_baselines(model: nn.Module, num_games: int = 20) -> None:
    model.eval()
    stats = {"win": 0, "draw": 0, "loss": 0}
    for game_idx in range(num_games):
        seed = 200000 + game_idx
        controlled_id = game_idx % 4
        opponents = build_opponents(controlled_id, seed)

        env = make_env(seed)
        obs = reset_env(env)
        done = False
        step = 0
        while not done:
            state = encode_features(obs["map"], obs["players"], obs["bombs"], controlled_id, step)
            with torch.no_grad():
                logits = model(torch.from_numpy(state).unsqueeze(0).to(DEVICE))[0]
                action = int(torch.argmax(logits).item())
            actions = [STOP, STOP, STOP, STOP]
            actions[controlled_id] = transform_action(action, controlled_id)
            for pid, opp in opponents.items():
                actions[pid] = int(opp.act(obs))
            obs, terminated, truncated = step_env(env, actions)
            done = terminated or truncated
            step += 1

        alive = [int(p[2]) for p in np.asarray(obs["players"])]
        my_alive = alive[controlled_id] if controlled_id < len(alive) else 0
        alive_count = sum(alive)
        if my_alive == 1 and alive_count == 1:
            stats["win"] += 1
        elif my_alive == 1:
            stats["draw"] += 1
        else:
            stats["loss"] += 1
    print(f"Quick eval proxy: {stats}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    print("=== Phase 1: Collect initial demonstrations ===", flush=True)
    collect_initial_data(INITIAL_GAMES)

    print("=== Phase 2: Train initial policy ===", flush=True)
    model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=None, lr=LEARNING_RATE)

    for round_idx in range(DAGGER_ROUNDS):
        print(f"=== Phase 3.{round_idx + 1}: DAgger collection ===", flush=True)
        new_samples = collect_dagger_data(model, TRAIN_DIR, DAGGER_GAMES_PER_ROUND)
        print(f"DAgger round {round_idx + 1}: collected {new_samples} samples", flush=True)

        print(f"=== Phase 4.{round_idx + 1}: Fine-tune with aggregated data ===", flush=True)
        model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_PATH, lr=FINE_TUNE_LR)

    print("=== Optional quick sanity check ===", flush=True)
    quick_eval_against_baselines(model, num_games=20)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
