import os
import sys
import json
import random
from dataclasses import dataclass
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.append(os.getcwd())

try:
    from engine.game import BomberEnv
except Exception as e:  # pragma: no cover
    BomberEnv = None
    _ENGINE_IMPORT_ERROR = repr(e)
else:
    _ENGINE_IMPORT_ERROR = None

# =============================================================================
# Config
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BOARD_SIZE = 13
NUM_ACTIONS = 6
MAX_STEPS = 500

SPATIAL_CHANNELS = 18
SCALAR_DIM = 10

INITIAL_GAMES = 600
DAGGER_ROUNDS = 2
DAGGER_GAMES_PER_ROUND = 180

PPO_ROUNDS = 6
PPO_GAMES_PER_ROUND = 200
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_CLIP_EPS = 0.20
PPO_GAMMA = 0.995
PPO_LAMBDA = 0.95
PPO_VALUE_COEF = 0.50
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 1.0

TRAIN_SPLIT_MOD = 10
CHUNK_SIZE = 2048
BATCH_SIZE = 128
EPOCHS_BC = 16
LEARNING_RATE_BC = 1e-3
LEARNING_RATE_PPO = 2.5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

TRAIN_DIR = "bc_train_chunks"
VAL_DIR = "bc_val_chunks"
MODEL_BC_PATH = "model_bc.pth"
MODEL_BC_BEST_PATH = "model_bc_best.pth"
MODEL_AC_PATH = "model_ac.pth"
MODEL_AC_BEST_PATH = "model_ac_best.pth"
MANIFEST_NAME = "manifest.json"
LEAGUE_DIR = "league_snapshots"

AUGMENT_FLIP_PROB = 0.50

R_STEP_ALIVE = 0.002
R_STEP_PENALTY = -0.004
R_BOX = 0.015
R_ITEM = 0.06
R_KILL = 0.60
R_DEATH = -4.0
R_WIN = 8.0
R_DRAW_ALIVE = 0.75
R_STOP_PENALTY = -0.002
R_SAFE_BOMB_BONUS = 0.04
R_UNSAFE_BOMB_PENALTY = -0.18

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
MOVES = {
    0: (0, 0),
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
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

def explosion_time_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, horizon: float = 8.0) -> np.ndarray:
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
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger
    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / 8.0
    danger[plane <= threshold] = 1.0
    return danger

def immediate_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    return danger_plane(grid, players, bombs, timer_threshold=1)

def chain_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, chain_horizon: int = 3) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane
    original = np.array([max(0, int(b[2])) for b in bombs], dtype=np.int32)
    effective = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
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

def future_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, horizon: float = 8.0) -> np.ndarray:
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

def bfs_distance_to_targets(grid: np.ndarray, start: Tuple[int, int], targets: set, bombs: np.ndarray, max_depth: int = 64) -> Optional[int]:
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
    return [int(a) for a in legal if int(a) in (1, 2, 3, 4)]

def _safe_escape_exists(grid: np.ndarray, start: Tuple[int, int], players: np.ndarray, bombs: np.ndarray, blocked: set, extra_bomb: Optional[Tuple[int, int, int, int]] = None, max_depth: int = 16) -> bool:
    augmented = bombs
    if extra_bomb is not None:
        extra = np.array([[extra_bomb[0], extra_bomb[1], extra_bomb[2], extra_bomb[3]]], dtype=np.int8)
        augmented = extra if bombs is None or len(bombs) == 0 else np.concatenate([bombs, extra], axis=0)
    deadlines = np.full((BOARD_SIZE, BOARD_SIZE), 10**9, dtype=np.int32)
    if augmented is not None and len(augmented) > 0:
        times = bomb_effective_explosion_times(grid, players, augmented)
        for i in range(len(augmented)):
            owner = int(augmented[i][3]) if augmented.shape[1] > 3 else -1
            radius = bomb_radius_for_owner(players, owner)
            t = int(max(0, times[i]))
            for r, c in blast_tiles(grid, int(augmented[i][0]), int(augmented[i][1]), radius):
                deadlines[r, c] = min(deadlines[r, c], t)

    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, dist = q.popleft()
        if dist > 0 and dist < int(deadlines[pos[0], pos[1]]):
            return True
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
            arr = dist + 1
            if arr >= int(deadlines[npos[0], npos[1]]):
                continue
            seen.add(npos)
            q.append((npos, arr))
    return False

def safe_to_bomb_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane
    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    bomb_radius = 1 + int(players[my_id][4])
    blocked = bomb_positions_set(bombs)
    imm = immediate_danger_plane(grid, players, bombs)
    chain = chain_danger_plane(grid, players, bombs)
    danger = np.maximum(imm, chain)
    enemy_positions = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(4)
        if i != my_id and i < len(players) and int(players[i][2]) == 1
    }
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not passable(grid, r, c):
                continue
            if (r, c) in blocked:
                continue
            if danger[r, c] > 0.0:
                continue
            blast = blast_tiles(grid, r, c, bomb_radius)
            hit_boxes = any(int(grid[x, y]) == 2 for x, y in blast)
            hit_enemy = any((x, y) in enemy_positions for x, y in blast)
            if not (hit_boxes or hit_enemy):
                continue
            exits = 0
            for a in (1, 2, 3, 4):
                nr, nc = next_pos((r, c), a)
                if not passable(grid, nr, nc):
                    continue
                if (nr, nc) in blocked:
                    continue
                if (nr, nc) in blast:
                    continue
                if danger[nr, nc] > 0.0:
                    continue
                exits += 1
            if exits > 0:
                plane[r, c] = 1.0
    return plane

# =============================================================================
# Observation encoding
# =============================================================================
def encode_obs(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    spatial = np.zeros((SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    spatial[0] = (grid == 1).astype(np.float32)
    spatial[1] = (grid == 2).astype(np.float32)
    spatial[2] = (grid == 0).astype(np.float32)
    spatial[3] = (grid == 3).astype(np.float32)
    spatial[4] = (grid == 4).astype(np.float32)

    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                spatial[5 + pid, r, c] = 1.0

    spatial[9] = explosion_time_plane(grid, players, bombs)
    spatial[10] = immediate_danger_plane(grid, players, bombs)
    spatial[11] = chain_danger_plane(grid, players, bombs)
    spatial[12] = future_danger_plane(grid, players, bombs)

    my_pos = (0, 0)
    bombs_left = 0
    bomb_radius = 1
    me_alive = 0
    if my_id < len(players) and int(players[my_id][2]) == 1:
        me_alive = 1
        mr, mc = int(players[my_id][0]), int(players[my_id][1])
        my_pos = (mr, mc)
        spatial[13, mr, mc] = 1.0
        bombs_left = int(players[my_id][3])
        bomb_radius = 1 + int(players[my_id][4])

    spatial[14] = safe_to_bomb_plane(grid, players, bombs, my_id)

    if bombs is not None and len(bombs) > 0:
        eff_times = bomb_effective_explosion_times(grid, players, bombs)
        for i, b in enumerate(bombs):
            r, c = int(b[0]), int(b[1])
            if not in_bounds(r, c):
                continue
            spatial[15, r, c] = max(spatial[15, r, c], 1.0 / float(max(int(eff_times[i]), 1)))
            owner = int(b[3]) if len(b) > 3 else -1
            spatial[16, r, c] = max(spatial[16, r, c], normalize_scalar(bomb_radius_for_owner(players, owner), 6.0))
    else:
        spatial[15].fill(0.0)
        spatial[16].fill(0.0)

    enemy_pos = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(4)
        if i != my_id and i < len(players) and int(players[i][2]) == 1
    }
    contested = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not passable(grid, r, c):
                continue
            d = min((abs(r - er) + abs(c - ec) for er, ec in enemy_pos), default=99)
            contested[r, c] = 1.0 - min(d, 8) / 8.0 if enemy_pos else 0.0
    spatial[17] = contested

    item_pos = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
    d_item = bfs_distance_to_targets(grid, my_pos, item_pos, bombs) if me_alive else None
    d_enemy = bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs) if me_alive else None
    reach = bfs_reachable_count(grid, my_pos, bombs, max_depth=3) if me_alive else 0

    scalar = np.array([
        normalize_scalar(bombs_left, 5.0),
        normalize_scalar(bomb_radius, 5.0),
        norm_dist(d_item),
        norm_dist(d_enemy),
        normalize_scalar(reach, 20.0),
        float(1 if me_alive else 0),
        normalize_scalar(len(bombs) if bombs is not None else 0, 10.0),
        normalize_scalar(step, float(MAX_STEPS)),
        normalize_scalar(int(np.sum(grid == 2)), 40.0),
        normalize_scalar(int(np.sum(players[:, 2])) if len(players) else 0, 4.0),
    ], dtype=np.float32)

    return spatial, scalar

# =============================================================================
# Baseline agents
# =============================================================================
class _FallbackRuleAgent:
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
    box_farmer: float = 1.2
    simple: float = 0.8
    random: float = 0.2

class TeacherEnsemble:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.tactical = _maybe_make(TacticalRuleAgent, agent_id) if "TacticalRuleAgent" in globals() else _FallbackRuleAgent(agent_id)
        self.genius = _maybe_make(GeniusRuleAgent, agent_id) if "GeniusRuleAgent" in globals() else _FallbackRuleAgent(agent_id)
        self.smarter = _maybe_make(SmarterRuleAgent, agent_id) if "SmarterRuleAgent" in globals() else _FallbackRuleAgent(agent_id)
        self.box_farmer = _maybe_make(BoxFarmerAgent, agent_id) if "BoxFarmerAgent" in globals() else _FallbackRuleAgent(agent_id)
        self.simple = _maybe_make(SimpleRuleAgent, agent_id) if "SimpleRuleAgent" in globals() else _FallbackRuleAgent(agent_id)
        self.random = _maybe_make(RandomAgent, agent_id) if "RandomAgent" in globals() else _FallbackRuleAgent(agent_id)
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
        priority = [actions["tactical"], actions["genius"], actions["smarter"], actions["box_farmer"], actions["simple"], actions["random"]]
        best_score = max(score.values())
        candidates = [a for a, s in score.items() if abs(s - best_score) < 1e-9]
        for preferred in priority:
            if preferred in candidates:
                return int(preferred)
        return int(candidates[0])

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
        if danger[r, c] > 0.0:
            safe_acts = []
            for a in movement_actions_from_legal(legal):
                nr, nc = next_pos((r, c), a)
                if in_bounds(nr, nc) and danger[nr, nc] == 0.0:
                    safe_acts.append(a)
            if safe_acts:
                for key in ("tactical", "genius", "smarter", "simple", "box_farmer", "random"):
                    a = actions[key]
                    if a in safe_acts:
                        return int(a)
                return int(random.choice(safe_acts))
        box_count = int(np.sum(grid == 2))
        self.weights.box_farmer = 2.0 if box_count >= 18 else 1.1
        alive_cnt = int(np.sum(players[:, 2])) if len(players) else 0
        if alive_cnt <= 2:
            self.weights.tactical = 3.5
            self.weights.genius = 2.8
        else:
            self.weights.tactical = 3.0
            self.weights.genius = 2.5
        return self._weighted_vote(actions, legal=legal)

# Try local imports, if available.
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
# Models
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return torch.relu(x)

class ActorCriticNet(nn.Module):
    def __init__(self, spatial_channels: int = SPATIAL_CHANNELS, scalar_dim: int = SCALAR_DIM, num_actions: int = NUM_ACTIONS, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(spatial_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.08),
            ResidualBlock(width, dropout=0.08),
            ResidualBlock(width, dropout=0.08),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(4)
        self.max_pool = nn.AdaptiveMaxPool2d(4)
        spatial_out = width * 4 * 4 * 2
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        trunk_in = spatial_out + 32
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(spatial)
        x = self.blocks(x)
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        s = self.scalar_mlp(scalars)
        h = torch.cat([avg_feat, max_feat, s], dim=1)
        h = self.trunk(h)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

# =============================================================================
# Dataset / storage
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

def flush_chunk(chunk_dir: str, chunk_idx: int, spatial_states: List[np.ndarray], scalar_states: List[np.ndarray], actions: List[int], seeds: List[int]) -> Dict:
    if not spatial_states:
        return {}
    spatial_np = np.stack(spatial_states, axis=0).astype(np.float32)
    scalar_np = np.stack(scalar_states, axis=0).astype(np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    seeds_np = np.array(seeds, dtype=np.int64)
    hist = np.bincount(actions_np, minlength=NUM_ACTIONS).astype(int).tolist()
    filename = f"chunk_{chunk_idx:05d}.npz"
    np.savez_compressed(os.path.join(chunk_dir, filename), spatial=spatial_np, scalars=scalar_np, actions=actions_np, seeds=seeds_np)
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
        chunk_indices = chunk_indices[worker_id::num_workers]
        for chunk_idx in chunk_indices:
            chunk_meta = self.chunks[int(chunk_idx)]
            data = np.load(os.path.join(self.chunk_dir, chunk_meta["file"]))
            spatial = data["spatial"]
            scalars = data["scalars"]
            actions = data["actions"]
            idxs = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(idxs)
            for i in idxs:
                s = torch.from_numpy(spatial[int(i)]).float()
                z = torch.from_numpy(scalars[int(i)]).float()
                a = int(actions[int(i)])
                if self.augment:
                    s, z, a = augment_tensor_and_action(s, z, a)
                yield s, z, torch.tensor(a, dtype=torch.long)

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

def augment_tensor_and_action(spatial: torch.Tensor, scalars: torch.Tensor, action: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if random.random() > AUGMENT_FLIP_PROB:
        return spatial, scalars, int(action)
    p = random.random()
    if p < 0.33:
        spatial = torch.flip(spatial, dims=[2])
        action = {1: 2, 2: 1, 3: 3, 4: 4, 0: 0, 5: 5}.get(int(action), int(action))
    elif p < 0.66:
        spatial = torch.flip(spatial, dims=[1])
        action = {3: 4, 4: 3, 1: 1, 2: 2, 0: 0, 5: 5}.get(int(action), int(action))
    else:
        spatial = torch.flip(spatial, dims=[1, 2])
        action = {1: 2, 2: 1, 3: 4, 4: 3, 0: 0, 5: 5}.get(int(action), int(action))
    return spatial, scalars, int(action)

# =============================================================================
# Safety shielding
# =============================================================================
def shielded_action_mask(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int, allow_bomb: bool = True) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        mask[0] = True
        return mask

    r, c = int(players[my_id][0]), int(players[my_id][1])
    bombs_left = int(players[my_id][3])
    blocked = bomb_positions_set(bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
    danger_future = future_danger_plane(grid, players, bombs)

    mask[0] = True
    for a in (1, 2, 3, 4):
        nr, nc = next_pos((r, c), a)
        if not passable(grid, nr, nc):
            continue
        if (nr, nc) in blocked:
            continue
        if danger_now[r, c] > 0.0 and danger_now[nr, nc] > 0.0:
            continue
        if danger_now[nr, nc] > 0.0 and danger_future[nr, nc] >= danger_future[r, c]:
            continue
        mask[a] = True

    if allow_bomb and bombs_left > 0 and (r, c) not in blocked:
        extra_bomb = (r, c, 7, my_id)
        if _safe_escape_exists(grid, (r, c), players, bombs, blocked, extra_bomb=extra_bomb, max_depth=16):
            mask[5] = True
    return mask

def masked_logits(logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    mask_t = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
    out = logits.clone()
    out[~mask_t] = -1e9
    if not bool(mask_t.any()):
        out[:] = 0.0
        out[0] = 0.0
    return out

def sample_action_from_logits(logits: torch.Tensor, mask: np.ndarray) -> Tuple[int, float, float]:
    masked = masked_logits(logits, mask)
    probs = torch.softmax(masked, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = int(dist.sample().item())
    log_prob = float(dist.log_prob(torch.tensor(action, device=logits.device)).item())
    entropy = float(dist.entropy().item())
    return action, log_prob, entropy

# =============================================================================
# Reward shaping
# =============================================================================
def count_boxes(grid: np.ndarray) -> int:
    return int(np.sum(grid == 2))

def alive_count(players: np.ndarray) -> int:
    return int(np.sum(players[:, 2])) if len(players) else 0

def compute_terminal_rank_reward(players: np.ndarray, my_id: int) -> float:
    if my_id >= len(players):
        return 0.0
    me_alive = int(players[my_id][2]) == 1
    alive_n = alive_count(players)
    if me_alive and alive_n == 1:
        return R_WIN
    if me_alive and alive_n > 1:
        return R_DRAW_ALIVE
    if not me_alive:
        return R_DEATH
    return 0.0

def compute_step_reward(prev_obs: Dict, obs: Dict, my_id: int, action: int) -> float:
    prev_grid = prev_obs["map"]
    new_grid = obs["map"]
    prev_players = prev_obs["players"]
    new_players = obs["players"]

    reward = 0.0
    me_alive_prev = int(prev_players[my_id][2]) == 1 if my_id < len(prev_players) else False
    me_alive_new = int(new_players[my_id][2]) == 1 if my_id < len(new_players) else False

    if me_alive_new:
        reward += R_STEP_ALIVE
    reward += R_STEP_PENALTY

    if action == 0:
        reward += R_STOP_PENALTY

    prev_boxes = count_boxes(prev_grid)
    new_boxes = count_boxes(new_grid)
    reward += R_BOX * max(0, prev_boxes - new_boxes)

    if my_id < len(prev_players) and my_id < len(new_players):
        prev_bonus = int(prev_players[my_id][4])
        new_bonus = int(new_players[my_id][4])
        prev_left = int(prev_players[my_id][3])
        new_left = int(new_players[my_id][3])
        if new_bonus > prev_bonus:
            reward += R_ITEM * (new_bonus - prev_bonus)
        if new_left > prev_left:
            reward += R_ITEM * 0.75 * (new_left - prev_left)

    prev_alive = alive_count(prev_players)
    new_alive = alive_count(new_players)
    if new_alive < prev_alive:
        reward += R_KILL * (prev_alive - new_alive)

    if action == 5 and me_alive_prev:
        grid = prev_obs["map"]
        players = prev_obs["players"]
        bombs = prev_obs["bombs"]
        blocked = bomb_positions_set(bombs)
        extra_bomb = (int(players[my_id][0]), int(players[my_id][1]), 7, my_id)
        if _safe_escape_exists(grid, (int(players[my_id][0]), int(players[my_id][1])), players, bombs, blocked, extra_bomb=extra_bomb, max_depth=16):
            reward += R_SAFE_BOMB_BONUS
        else:
            reward += R_UNSAFE_BOMB_PENALTY

    if me_alive_prev and not me_alive_new:
        reward += R_DEATH

    return float(np.clip(reward, -6.0, 10.0))

# =============================================================================
# Rollout storage
# =============================================================================
@dataclass
class RolloutStep:
    spatial: np.ndarray
    scalars: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool
    seed: int

# =============================================================================
# Opponent setup
# =============================================================================
def build_opponents(controlled_id: int, game_seed: int, league_models: Optional[List[str]] = None) -> Dict[int, object]:
    rng = random.Random(game_seed)
    base_pool = [cls for cls in [TacticalRuleAgent, GeniusRuleAgent, SmarterRuleAgent, BoxFarmerAgent, SimpleRuleAgent, RandomAgent] if cls is not None]
    if not base_pool:
        base_pool = [_FallbackRuleAgent]

    other_ids = [pid for pid in range(4) if pid != controlled_id]
    opponents: Dict[int, object] = {}
    chosen = [rng.choice(base_pool) for _ in range(3)]
    for pid, cls in zip(other_ids, chosen):
        opponents[pid] = cls(pid)

    if league_models:
        snap_path = rng.choice(league_models)
        if os.path.exists(snap_path):
            try:
                opponents[other_ids[0]] = FrozenPolicyAgent(other_ids[0], snap_path)
            except Exception:
                pass
    return opponents

class FrozenPolicyAgent:
    def __init__(self, agent_id: int, model_path: str):
        self.agent_id = int(agent_id)
        self.model = ActorCriticNet().to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def act(self, obs: Dict) -> int:
        step = int(obs.get("step", 0))
        spatial, scalars = encode_obs(obs["map"], obs["players"], obs["bombs"], self.agent_id, step)
        mask = shielded_action_mask(obs["map"], obs["players"], obs["bombs"], self.agent_id, step, allow_bomb=True)
        with torch.no_grad():
            s = torch.from_numpy(spatial).unsqueeze(0).to(DEVICE)
            z = torch.from_numpy(scalars).unsqueeze(0).to(DEVICE)
            logits, _ = self.model(s, z)
            action = int(torch.argmax(masked_logits(logits[0], mask)).item())
        return action

# =============================================================================
# BC data collection / training
# =============================================================================
def collect_one_step(obs: Dict, teacher: TeacherEnsemble, my_id: int, step: int) -> Tuple[np.ndarray, np.ndarray, int]:
    spatial, scalars = encode_obs(obs["map"], obs["players"], obs["bombs"], my_id, step)
    action = int(teacher.act(obs))
    return spatial, scalars, action

def collect_initial_data(train_dir: str, val_dir: str, num_games: int) -> None:
    ensure_dir(train_dir)
    ensure_dir(val_dir)
    train_manifest = load_manifest(train_dir)
    val_manifest = load_manifest(val_dir)
    train_chunk_idx = len(train_manifest.get("chunks", []))
    val_chunk_idx = len(val_manifest.get("chunks", []))
    train_spatial, train_scalars, train_actions, train_seeds = [], [], [], []
    val_spatial, val_scalars, val_actions, val_seeds = [], [], [], []

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
            s, z, a = collect_one_step(obs, teacher, controlled_id, step)
            if split == "train":
                train_spatial.append(s); train_scalars.append(z); train_actions.append(a); train_seeds.append(seed)
            else:
                val_spatial.append(s); val_scalars.append(z); val_actions.append(a); val_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = a
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))
            obs = dict(obs)
            obs["step"] = step
            nxt, terminated, truncated = env.step(actions)
            obs = nxt
            done = bool(terminated or truncated)
            step += 1

            if split == "train" and len(train_spatial) >= CHUNK_SIZE:
                entry = flush_chunk(train_dir, train_chunk_idx, train_spatial, train_scalars, train_actions, train_seeds)
                if entry:
                    train_manifest["chunks"].append(entry)
                    save_manifest(train_dir, train_manifest)
                    train_chunk_idx += 1
                train_spatial.clear(); train_scalars.clear(); train_actions.clear(); train_seeds.clear()

            if split == "val" and len(val_spatial) >= CHUNK_SIZE:
                entry = flush_chunk(val_dir, val_chunk_idx, val_spatial, val_scalars, val_actions, val_seeds)
                if entry:
                    val_manifest["chunks"].append(entry)
                    save_manifest(val_dir, val_manifest)
                    val_chunk_idx += 1
                val_spatial.clear(); val_scalars.clear(); val_actions.clear(); val_seeds.clear()

        if (game_idx + 1) % 100 == 0:
            tr = sum(c["count"] for c in train_manifest.get("chunks", [])) + len(train_actions)
            va = sum(c["count"] for c in val_manifest.get("chunks", [])) + len(val_actions)
            print(f"Collected {game_idx + 1}/{num_games} games | train={tr} | val={va}", flush=True)

    if train_spatial:
        entry = flush_chunk(train_dir, train_chunk_idx, train_spatial, train_scalars, train_actions, train_seeds)
        if entry:
            train_manifest["chunks"].append(entry)
    if val_spatial:
        entry = flush_chunk(val_dir, val_chunk_idx, val_spatial, val_scalars, val_actions, val_seeds)
        if entry:
            val_manifest["chunks"].append(entry)
    save_manifest(train_dir, train_manifest)
    save_manifest(val_dir, val_manifest)

def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int, league_models: Optional[List[str]] = None, round_idx: int = 0) -> int:
    ensure_dir(out_dir)
    model.eval()
    out_manifest = load_manifest(out_dir)
    chunk_idx = len(out_manifest.get("chunks", []))
    buf_spatial, buf_scalars, buf_actions, buf_seeds = [], [], [], []
    collected = 0

    def flush_buffer():
        nonlocal chunk_idx, collected
        if not buf_spatial:
            return
        entry = flush_chunk(out_dir, chunk_idx, buf_spatial, buf_scalars, buf_actions, buf_seeds)
        if entry:
            out_manifest["chunks"].append(entry)
            save_manifest(out_dir, out_manifest)
            collected += entry["count"]
            chunk_idx += 1
        buf_spatial.clear(); buf_scalars.clear(); buf_actions.clear(); buf_seeds.clear()

    for game_idx in range(num_games):
        seed = 100000 + SEED + round_idx * 10000 + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherEnsemble(controlled_id)
        opponents = build_opponents(controlled_id, seed, league_models=league_models)
        done = False
        step = 0
        while not done:
            spatial, scalars = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step)
            with torch.no_grad():
                s = torch.from_numpy(spatial).unsqueeze(0).to(DEVICE)
                z = torch.from_numpy(scalars).unsqueeze(0).to(DEVICE)
                logits, _ = model(s, z)
                student_action = int(torch.argmax(logits, dim=1).item())
            expert_action = int(teacher.act(obs))
            if student_action != expert_action or (student_action == 0 and expert_action != 0):
                buf_spatial.append(spatial)
                buf_scalars.append(scalars)
                buf_actions.append(expert_action)
                buf_seeds.append(seed)
            actions = [0, 0, 0, 0]
            actions[controlled_id] = student_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))
            obs = dict(obs)
            obs["step"] = step
            nxt, terminated, truncated = env.step(actions)
            obs = nxt
            done = bool(terminated or truncated)
            step += 1
            if len(buf_spatial) >= CHUNK_SIZE:
                flush_buffer()
        if (game_idx + 1) % 50 == 0:
            print(f"DAgger {game_idx + 1}/{num_games} games | new samples={collected + len(buf_actions)}", flush=True)

    flush_buffer()
    return collected

def build_loaders(train_dir: str, val_dir: str):
    train_ds = ChunkedBomberIterableDataset(train_dir, augment=True, shuffle_chunks=True, shuffle_within_chunk=True, seed=SEED)
    val_ds = ChunkedBomberIterableDataset(val_dir, augment=False, shuffle_chunks=False, shuffle_within_chunk=False, seed=SEED)
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation samples found in {val_dir}")
    loader_kwargs = dict(batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(DEVICE.type == "cuda"), drop_last=False)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    class_weights = compute_class_weights(train_dir).to(DEVICE)
    return train_loader, val_loader, class_weights

def run_epoch_bc(model: nn.Module, loader: DataLoader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_idx, (spatial, scalars, actions) in enumerate(loader):
        spatial = spatial.to(DEVICE, non_blocking=True)
        scalars = scalars.to(DEVICE, non_blocking=True)
        actions = actions.to(DEVICE, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits, _ = model(spatial, scalars)
        loss = criterion(logits, actions)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * spatial.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == actions).sum().item())
        total_count += int(spatial.size(0))
        if batch_idx % 50 == 0:
            print(f"  batch={batch_idx}", flush=True)
    return total_loss / max(1, total_count), total_correct / max(1, total_count)

def train_bc_model(train_dir: str, val_dir: str, init_model_path: Optional[str] = None, lr: float = LEARNING_RATE_BC):
    train_loader, val_loader, class_weights = build_loaders(train_dir, val_dir)
    model = ActorCriticNet().to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        state = torch.load(init_model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE
    for epoch in range(1, EPOCHS_BC + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS_BC}", flush=True)
        train_loss, train_acc = run_epoch_bc(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_epoch_bc(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS_BC} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), MODEL_BC_BEST_PATH)
            patience_left = PATIENCE
            print(f"  -> saved best model to {MODEL_BC_BEST_PATH}", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("  -> early stopping", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_BC_PATH)
    print(f"Final model saved to {MODEL_BC_PATH}", flush=True)
    return model

# =============================================================================
# PPO / self-play
# =============================================================================
@dataclass
class RolloutStep:
    spatial: np.ndarray
    scalars: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool
    seed: int

def _ppo_gae(rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool], bootstrap_value: float = 0.0, truncated: bool = False, gamma: float = PPO_GAMMA, lam: float = PPO_LAMBDA) -> Tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = float(bootstrap_value if truncated else 0.0)
            next_nonterminal = 0.0 if dones[t] else 1.0
        else:
            next_value = float(values[t + 1])
            next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns

def _normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    adv = advantages.astype(np.float32)
    return (adv - adv.mean()) / (adv.std() + 1e-8)

def _evaluate_state(model: nn.Module, obs: Dict, my_id: int, step: int) -> Tuple[int, float]:
    spatial, scalars = encode_obs(obs["map"], obs["players"], obs["bombs"], my_id, step)
    with torch.no_grad():
        s = torch.from_numpy(spatial).unsqueeze(0).to(DEVICE)
        z = torch.from_numpy(scalars).unsqueeze(0).to(DEVICE)
        logits, value = model(s, z)
        return int(torch.argmax(logits, dim=1).item()), float(value.item())

def quick_eval_against_baselines(model: nn.Module, num_games: int = 20, round_idx: int = 0, league_models: Optional[List[str]] = None) -> None:
    model.eval()
    wins = draws = losses = 0
    total_steps = 0
    for game_idx in range(num_games):
        seed = 200000 + SEED + round_idx * 10000 + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_opponents(controlled_id, seed, league_models=league_models)
        done = False
        step = 0
        while not done:
            action, _ = _evaluate_state(model, obs, controlled_id, step)
            actions = [0, 0, 0, 0]
            actions[controlled_id] = action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))
            obs = dict(obs)
            obs["step"] = step
            nxt, terminated, truncated = env.step(actions)
            obs = nxt
            done = bool(terminated or truncated)
            step += 1
        total_steps += step
        alive = [int(p[2]) for p in obs["players"]]
        my_alive = alive[controlled_id]
        alive_count = sum(alive)
        if my_alive == 1 and alive_count == 1:
            wins += 1
        elif my_alive == 1:
            draws += 1
        else:
            losses += 1
    print(f"Quick eval proxy | wins={wins} draws={draws} losses={losses} | avg_survival_steps={total_steps / max(1, num_games):.1f}", flush=True)

def collect_selfplay_rollouts(model: nn.Module, num_games: int, round_idx: int, league_models: Optional[List[str]] = None) -> List[RolloutStep]:
    model.eval()
    rollouts: List[RolloutStep] = []
    for game_idx in range(num_games):
        seed = 300000 + SEED + round_idx * 10000 + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_opponents(controlled_id, seed, league_models=league_models)
        done = False
        step = 0
        while not done:
            spatial, scalars = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step)
            s = torch.from_numpy(spatial).unsqueeze(0).to(DEVICE)
            z = torch.from_numpy(scalars).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, value = model(s, z)
                mask = shielded_action_mask(obs["map"], obs["players"], obs["bombs"], controlled_id, step, allow_bomb=True)
                action, log_prob, _ = sample_action_from_logits(logits[0], mask)

            prev_obs = obs
            actions = [0, 0, 0, 0]
            actions[controlled_id] = action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))
            obs = dict(obs)
            obs["step"] = step
            nxt, terminated, truncated = env.step(actions)
            reward = compute_step_reward(prev_obs, nxt, controlled_id, action)
            done_flag = bool(terminated or truncated)
            rollouts.append(RolloutStep(
                spatial=spatial,
                scalars=scalars,
                action=action,
                log_prob=log_prob,
                value=float(value.item()),
                reward=reward,
                done=done_flag and bool(terminated),
                truncated=done_flag and bool(truncated),
                seed=seed,
            ))
            obs = nxt
            done = done_flag
            step += 1
        terminal_reward = compute_terminal_rank_reward(obs["players"], controlled_id)
        if rollouts:
            rollouts[-1].reward += terminal_reward
    return rollouts

def ppo_update(model: nn.Module, rollouts: List[RolloutStep], epochs: int = PPO_EPOCHS) -> None:
    if not rollouts:
        return
    spatial = torch.from_numpy(np.stack([r.spatial for r in rollouts])).float().to(DEVICE)
    scalars = torch.from_numpy(np.stack([r.scalars for r in rollouts])).float().to(DEVICE)
    actions = torch.tensor([r.action for r in rollouts], dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor([r.log_prob for r in rollouts], dtype=torch.float32, device=DEVICE)
    values = np.array([r.value for r in rollouts], dtype=np.float32)
    rewards = np.array([r.reward for r in rollouts], dtype=np.float32)
    dones = np.array([r.done for r in rollouts], dtype=np.bool_)
    truncated = bool(any(r.truncated for r in rollouts[-3:]))

    bootstrap_value = 0.0
    if truncated:
        with torch.no_grad():
            _, v = model(spatial[-1:].detach(), scalars[-1:].detach())
            bootstrap_value = float(v.item())

    advantages, returns = _ppo_gae(rewards, values, dones, bootstrap_value=bootstrap_value, truncated=truncated, gamma=PPO_GAMMA, lam=PPO_LAMBDA)
    advantages = _normalize_advantages(advantages)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_PPO, weight_decay=WEIGHT_DECAY)
    n = len(rollouts)
    idxs = np.arange(n)
    for epoch in range(1, epochs + 1):
        np.random.shuffle(idxs)
        total_loss = total_policy = total_value = total_entropy = 0.0
        steps = 0
        for start in range(0, n, PPO_MINIBATCH_SIZE):
            mb = idxs[start:start + PPO_MINIBATCH_SIZE]
            mb_spatial = spatial[mb]
            mb_scalars = scalars[mb]
            mb_actions = actions[mb]
            mb_old_log_probs = old_log_probs[mb]
            mb_returns = returns_t[mb]
            mb_adv = adv_t[mb]

            logits, values_pred = model(mb_spatial, mb_scalars)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * mb_adv
            policy_loss = -torch.mean(torch.min(unclipped, clipped))
            value_loss = torch.mean((values_pred - mb_returns) ** 2)
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            steps += 1
        print(f"PPO epoch {epoch:02d}/{epochs} | loss={total_loss/max(1,steps):.4f} | policy={total_policy/max(1,steps):.4f} | value={total_value/max(1,steps):.4f} | entropy={total_entropy/max(1,steps):.4f}", flush=True)

def save_league_snapshot(model: nn.Module, round_idx: int) -> str:
    ensure_dir(LEAGUE_DIR)
    path = os.path.join(LEAGUE_DIR, f"snapshot_round_{round_idx:02d}.pth")
    torch.save(model.state_dict(), path)
    return path

def train_selfplay_actor_critic(model: nn.Module, rounds: int = PPO_ROUNDS, games_per_round: int = PPO_GAMES_PER_ROUND, league_models: Optional[List[str]] = None):
    league_models = list(league_models or [])
    for round_idx in range(1, rounds + 1):
        print(f"=== Phase 5.{round_idx}: Self-play actor-critic fine-tuning ===", flush=True)
        quick_eval_against_baselines(model, num_games=20, round_idx=round_idx, league_models=league_models)
        rollouts = collect_selfplay_rollouts(model, num_games=games_per_round, round_idx=round_idx, league_models=league_models)
        print(f"Round {round_idx}: collected {len(rollouts)} steps", flush=True)
        ppo_update(model, rollouts, epochs=PPO_EPOCHS)
        snap = save_league_snapshot(model, round_idx)
        league_models.append(snap)
        if len(league_models) > 6:
            league_models.pop(0)
    return model

# =============================================================================
# Main
# =============================================================================
def main():
    if _ENGINE_IMPORT_ERROR is not None:
        raise RuntimeError(f"Failed to import BomberEnv: {_ENGINE_IMPORT_ERROR}")

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(LEAGUE_DIR)

    if BASELINE_IMPORT_ERRORS:
        print("Some baseline imports failed; trainer will use fallback rules where needed.", flush=True)
        for name, err in BASELINE_IMPORT_ERRORS:
            print(f"  - {name}: {err}", flush=True)

    print("=== Phase 1: Collect initial demonstrations ===", flush=True)
    collect_initial_data(TRAIN_DIR, VAL_DIR, INITIAL_GAMES)

    print("=== Phase 2: Train initial BC policy ===", flush=True)
    model = train_bc_model(TRAIN_DIR, VAL_DIR, init_model_path=None, lr=LEARNING_RATE_BC)

    league_models = [MODEL_BC_PATH]
    for round_idx in range(DAGGER_ROUNDS):
        print(f"=== Phase 3.{round_idx + 1}: DAgger collection ===", flush=True)
        new_samples = collect_dagger_data(model, TRAIN_DIR, DAGGER_GAMES_PER_ROUND, league_models=league_models, round_idx=round_idx)
        print(f"DAgger round {round_idx + 1}: collected {new_samples} correction samples", flush=True)
        print(f"=== Phase 4.{round_idx + 1}: BC fine-tune with aggregated data ===", flush=True)
        model = train_bc_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_BC_PATH, lr=LEARNING_RATE_BC * 0.4)
        torch.save(model.state_dict(), MODEL_AC_PATH)
        league_models.append(MODEL_AC_PATH)
        if len(league_models) > 6:
            league_models.pop(0)

    model = model.to(DEVICE)
    print("=== Phase 5: Self-play actor-critic fine-tuning ===", flush=True)
    model = train_selfplay_actor_critic(model, rounds=PPO_ROUNDS, games_per_round=PPO_GAMES_PER_ROUND, league_models=league_models)

    torch.save(model.state_dict(), MODEL_AC_BEST_PATH)
    print("=== Optional quick sanity check ===", flush=True)
    quick_eval_against_baselines(model, num_games=20, round_idx=999, league_models=league_models)
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
