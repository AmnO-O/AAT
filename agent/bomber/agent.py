
import os
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BOARD_SIZE = 13
MAX_STEPS = 500
NUM_ACTIONS = 6
INPUT_CHANNELS = 24

STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = range(6)
ACTION_DELTAS = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP: (-1, 0),
    DOWN: (1, 0),
}
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PASSABLE = {0, 3, 4}

# -----------------------------------------------------------------------------
# Geometry / transforms
# -----------------------------------------------------------------------------
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
        if not in_bounds(r, c) or (r, c) in blocked or not passable(grid, r, c):
            continue
        free_n = 0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if passable(grid, nr, nc) and (nr, nc) not in blocked:
                free_n += 1
        out[r, c] = 1.0 - float(free_n) / 4.0
    return out


def build_safe_reachability(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    danger = danger_map(grid, players, bombs, horizon=1)
    dist = bfs_distance_map(grid, start, bombs, max_depth=16)
    return ((dist >= 0) & (danger == 0.0)).astype(np.float32)


def can_escape_after_bomb(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bomb_radius: int) -> bool:
    future = danger_map(grid, players, bombs, horizon=1)
    unsafe = blast_tiles(grid, pos, bomb_radius)
    blocked = bomb_positions_set(bombs)
    q = deque([(pos[0], pos[1], 0)])
    seen = {pos}
    while q:
        r, c, d = q.popleft()
        if d > 0 and future[r, c] == 0.0 and (r, c) not in unsafe:
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
        if r == r0 and abs(c - c0) <= radius:
            step = 1 if c > c0 else -1
            clear = True
            for cc in range(c0 + step, c, step):
                if int(grid[r0, cc]) in (1, 2):
                    clear = False
                    break
            if clear:
                return True
        if c == c0 and abs(r - r0) <= radius:
            step = 1 if r > r0 else -1
            clear = True
            for rr in range(r0 + step, r, step):
                if int(grid[rr, c0]) in (1, 2):
                    clear = False
                    break
            if clear:
                return True
    return False

# -----------------------------------------------------------------------------
# Feature encoding
# -----------------------------------------------------------------------------
def encode_features(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, agent_id: int, step: int) -> np.ndarray:
    grid, players, bombs = canonicalize_obs({"map": grid, "players": players, "bombs": bombs}, agent_id)
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    alive, pos, bombs_left, bomb_radius = my_info(players, agent_id)
    if alive and in_bounds(*pos):
        state[5, pos[0], pos[1]] = 1.0

    enemy_positions = []
    for pid in range(len(players)):
        if pid == agent_id or int(players[pid][2]) != 1:
            continue
        r, c = int(players[pid][0]), int(players[pid][1])
        if in_bounds(r, c):
            state[6, r, c] = 1.0
            enemy_positions.append((r, c))

    if alive and enemy_positions:
        dist_map = bfs_distance_map(grid, pos, bombs, max_depth=64)
        best_enemy = None
        best_dist = 10**9
        for ep in enemy_positions:
            d = int(dist_map[ep[0], ep[1]])
            if d >= 0 and d < best_dist:
                best_dist = d
                best_enemy = ep
        if best_enemy is not None:
            state[7, best_enemy[0], best_enemy[1]] = 1.0

    if bombs is not None and len(bombs) > 0:
        times = bomb_explosion_times(grid, players, bombs)
        for i in range(len(bombs)):
            r, c = int(bombs[i][0]), int(bombs[i][1])
            if not in_bounds(r, c):
                continue
            owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
            state[8, r, c] = 1.0
            if owner == agent_id:
                state[9, r, c] = 1.0
            elif 0 <= owner < len(players):
                state[10, r, c] = 1.0
            state[11, r, c] = 1.0 / float(max(1, int(bombs[i][2]) + 1))

    state[12] = danger_map(grid, players, bombs, horizon=1)
    state[13] = danger_map(grid, players, bombs, horizon=3)

    if alive:
        state[14] = build_safe_reachability(grid, players, bombs, pos)
        can_bomb = 1.0 if (bombs_left > 0 and pos not in bomb_positions_set(bombs) and can_escape_after_bomb(grid, players, bombs, pos, bomb_radius)) else 0.0
        state[15, pos[0], pos[1]] = can_bomb
        deg = local_degree_map(grid, bombs)
        state[16] = (deg <= 0.25).astype(np.float32)

        dist_map = bfs_distance_map(grid, pos, bombs, max_depth=64)
        item_targets = [(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))]
        enemy_targets = enemy_positions
        best_item = min((int(dist_map[r, c]) for r, c in item_targets if int(dist_map[r, c]) >= 0), default=-1)
        best_enemy = min((int(dist_map[r, c]) for r, c in enemy_targets if int(dist_map[r, c]) >= 0), default=-1)
        state[17] = np.where(best_item >= 0, float(best_item) / 24.0, 1.0)
        state[18] = np.where(best_enemy >= 0, float(best_enemy) / 24.0, 1.0)
        state[19] = local_box_density(grid, radius=2)
        state[20] = enemy_trap_potential(grid, players, agent_id, bombs)
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
# Inference agent
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.device = torch.device("cpu")
        self.step = 0
        self.model = BomberNet().to(self.device)
        self.model.eval()
        self.rng = random.Random(2026 + self.agent_id)
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        candidates = [
            "model_bc_best.pth",
            "model_bc.pth",
            "weights.pth",
            os.path.join(os.getcwd(), "model_bc_best.pth"),
            os.path.join(os.getcwd(), "model_bc.pth"),
            os.path.join(os.getcwd(), "weights.pth"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state, strict=True)
                self.model.eval()
                return
            except Exception:
                continue
        self.model = None  # type: ignore

    def act(self, obs: dict) -> int:
        try:
            grid, players, bombs = canonicalize_obs(obs, self.agent_id)
            alive, pos, bombs_left, bomb_radius = my_info(players, self.agent_id)
            if not alive:
                self.step += 1
                return STOP

            danger_now = danger_map(grid, players, bombs, horizon=1)
            if danger_now[pos[0], pos[1]] > 0.0:
                escape = self._escape_action(grid, players, bombs, pos)
                self.step += 1
                if escape is not None:
                    return transform_action(escape, self.agent_id)
                return STOP

            legal = self._legal_actions(grid, bombs, pos, bombs_left)
            if self.model is None:
                action = self._rule_fallback(grid, players, bombs, pos, bombs_left, bomb_radius, legal)
                self.step += 1
                return transform_action(action, self.agent_id)

            state = encode_features(obs["map"], obs["players"], obs["bombs"], self.agent_id, self.step)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device))[0].cpu().numpy()

            action = self._select_action(grid, players, bombs, pos, bombs_left, bomb_radius, logits, legal)
            self.step += 1
            return transform_action(action, self.agent_id)
        except Exception:
            self.step += 1
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

    def _count_boxes_in_blast(self, grid: np.ndarray, pos: Tuple[int, int], radius: int) -> int:
        return sum(1 for r, c in blast_tiles(grid, pos, radius) if int(grid[r, c]) == 2)

    def _can_hit_enemy(self, grid: np.ndarray, players: np.ndarray, pos: Tuple[int, int], radius: int) -> bool:
        r0, c0 = pos
        for pid in range(len(players)):
            if pid == self.agent_id or int(players[pid][2]) != 1:
                continue
            r, c = int(players[pid][0]), int(players[pid][1])
            if r == r0 and abs(c - c0) <= radius:
                step = 1 if c > c0 else -1
                clear = True
                for cc in range(c0 + step, c, step):
                    if int(grid[r0, cc]) in (1, 2):
                        clear = False
                        break
                if clear:
                    return True
            if c == c0 and abs(r - r0) <= radius:
                step = 1 if r > r0 else -1
                clear = True
                for rr in range(r0 + step, r, step):
                    if int(grid[rr, c0]) in (1, 2):
                        clear = False
                        break
                if clear:
                    return True
        return False

    def _can_escape_after_bomb(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bomb_radius: int) -> bool:
        return can_escape_after_bomb(grid, players, bombs, pos, bomb_radius)

    def _select_action(
        self,
        grid: np.ndarray,
        players: np.ndarray,
        bombs: np.ndarray,
        pos: Tuple[int, int],
        bombs_left: int,
        bomb_radius: int,
        logits: np.ndarray,
        legal: List[int],
    ) -> int:
        scores = np.full((NUM_ACTIONS,), -1e9, dtype=np.float32)
        for a in legal:
            scores[a] = float(logits[a])

        danger_now = danger_map(grid, players, bombs, horizon=1)
        blocked = bomb_positions_set(bombs)
        dist_map = bfs_distance_map(grid, pos, bombs, max_depth=64)
        item_targets = [(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))]
        enemy_targets = [(int(players[pid][0]), int(players[pid][1])) for pid in range(len(players)) if pid != self.agent_id and int(players[pid][2]) == 1]

        # Hard safety constraints.
        for a in [LEFT, RIGHT, UP, DOWN]:
            if a not in legal:
                continue
            nxt = move_from(pos, a)
            if danger_now[nxt[0], nxt[1]] > 0.0:
                scores[a] -= 10.0

        # Small preference for progress and open space.
        for a in [LEFT, RIGHT, UP, DOWN]:
            if a not in legal:
                continue
            nxt = move_from(pos, a)
            if item_targets:
                cur_item = min((int(dist_map[r, c]) for r, c in item_targets if int(dist_map[r, c]) >= 0), default=999)
                nd = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                nxt_item = min((int(nd[r, c]) for r, c in item_targets if int(nd[r, c]) >= 0), default=999)
                if nxt_item < cur_item:
                    scores[a] += 0.15 * (cur_item - nxt_item)
            if enemy_targets:
                cur_enemy = min((int(dist_map[r, c]) for r, c in enemy_targets if int(dist_map[r, c]) >= 0), default=999)
                nd = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                nxt_enemy = min((int(nd[r, c]) for r, c in enemy_targets if int(nd[r, c]) >= 0), default=999)
                if nxt_enemy < cur_enemy:
                    scores[a] += 0.08 * (cur_enemy - nxt_enemy)
            open_n = 0
            for dr, dc in DIRECTIONS:
                nr, nc = nxt[0] + dr, nxt[1] + dc
                if passable(grid, nr, nc) and (nr, nc) not in blocked:
                    open_n += 1
            scores[a] += 0.03 * open_n

        if PLACE_BOMB in legal:
            boxes = self._count_boxes_in_blast(grid, pos, bomb_radius)
            enemy_hit = self._can_hit_enemy(grid, players, pos, bomb_radius)
            escape_ok = self._can_escape_after_bomb(grid, players, bombs, pos, bomb_radius)
            bomb_score = scores[PLACE_BOMB]
            bomb_score += boxes * 0.7
            if enemy_hit:
                bomb_score += 2.2
            if escape_ok:
                bomb_score += 0.8
            else:
                bomb_score -= 2.0
            if boxes == 0 and not enemy_hit:
                bomb_score -= 0.6
            scores[PLACE_BOMB] = bomb_score

        # Prefer movement over waiting when something useful exists.
        if STOP in legal and any(scores[a] > scores[STOP] for a in [LEFT, RIGHT, UP, DOWN] if a in legal):
            scores[STOP] -= 0.2

        # Deterministic greedy choice with tiny seeded jitter for tie-breaks.
        noise = np.array([self.rng.random() * 1e-6 for _ in range(NUM_ACTIONS)], dtype=np.float32)
        best = int(np.argmax(scores + noise))
        if best not in legal:
            best = max(legal, key=lambda a: scores[a])
        return int(best)

    def _rule_fallback(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bombs_left: int, bomb_radius: int, legal: List[int]) -> int:
        danger_now = danger_map(grid, players, bombs, horizon=1)
        if danger_now[pos[0], pos[1]] > 0.0:
            escape = self._escape_action(grid, players, bombs, pos)
            if escape is not None:
                return int(escape)

        dist_map = bfs_distance_map(grid, pos, bombs, max_depth=64)
        item_targets = [(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))]
        enemy_targets = [(int(players[pid][0]), int(players[pid][1])) for pid in range(len(players)) if pid != self.agent_id and int(players[pid][2]) == 1]
        blocked = bomb_positions_set(bombs)

        move_scores = {}
        for a in [LEFT, RIGHT, UP, DOWN]:
            if a not in legal:
                continue
            nxt = move_from(pos, a)
            if danger_now[nxt[0], nxt[1]] > 0.0:
                continue
            score = 0.0
            if item_targets:
                cur_item = min((int(dist_map[r, c]) for r, c in item_targets if int(dist_map[r, c]) >= 0), default=999)
                nd = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                nxt_item = min((int(nd[r, c]) for r, c in item_targets if int(nd[r, c]) >= 0), default=999)
                score += max(0, cur_item - nxt_item) * 0.2
            if enemy_targets:
                cur_enemy = min((int(dist_map[r, c]) for r, c in enemy_targets if int(dist_map[r, c]) >= 0), default=999)
                nd = bfs_distance_map(grid, nxt, bombs, max_depth=64)
                nxt_enemy = min((int(nd[r, c]) for r, c in enemy_targets if int(nd[r, c]) >= 0), default=999)
                score += max(0, cur_enemy - nxt_enemy) * 0.08
            open_n = sum(1 for dr, dc in DIRECTIONS if passable(grid, nxt[0] + dr, nxt[1] + dc) and (nxt[0] + dr, nxt[1] + dc) not in blocked)
            score += 0.03 * open_n
            move_scores[a] = score

        if bombs_left > 0 and PLACE_BOMB in legal:
            boxes = self._count_boxes_in_blast(grid, pos, bomb_radius)
            enemy_hit = self._can_hit_enemy(grid, players, pos, bomb_radius)
            if (boxes > 0 or enemy_hit) and self._can_escape_after_bomb(grid, players, bombs, pos, bomb_radius):
                return PLACE_BOMB

        if move_scores:
            return max(move_scores.items(), key=lambda kv: kv[1])[0]
        return STOP
