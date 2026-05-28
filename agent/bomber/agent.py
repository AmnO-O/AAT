"""
Bomberland Agent – Neural Policy with Safety Escape (24‑channel encoder)
======================================================================
Tương thích với script huấn luyện đã cung cấp.
Sử dụng mạng neural đã huấn luyện để chọn hành động.
Nếu mạng chọn nước đi vào vùng nguy hiểm tức thời, BFS sẽ tìm lối thoát an toàn.
"""

import os
import random
import math
from collections import deque
from typing import List, Tuple, Set, Optional

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STOP    = 0
LEFT    = 1
RIGHT   = 2
UP      = 3
DOWN    = 4
BOMB    = 5

MOVES = {
    STOP:  (0, 0),
    LEFT:  (0, -1),
    RIGHT: (0, 1),
    UP:    (-1, 0),
    DOWN:  (1, 0),
}

BOARD_SIZE      = 13
INPUT_CHANNELS  = 24
MAX_STEPS       = 500
EXPLOSION_TIME_HORIZON = 8.0   # must match training

# -----------------------------------------------------------------------------
# Model definition (identical to training)
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

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
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_actions: int = 6, width: int = 64):
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
            ResidualBlock(width, dropout=0.05),
            ResidualBlock(width, dropout=0.05),
            ResidualBlock(width, dropout=0.05),
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


# -----------------------------------------------------------------------------
# Utility functions (copied from training script)
# -----------------------------------------------------------------------------
def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)


def next_pos(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    dr, dc = MOVES[int(action)]
    return pos[0] + dr, pos[1] + dc


def bomb_positions_set(bombs: np.ndarray) -> Set[Tuple[int, int]]:
    if bombs is None or len(bombs) == 0:
        return set()
    return {(int(b[0]), int(b[1])) for b in bombs}


def bomb_radius_for_owner(players: np.ndarray, owner: int) -> int:
    if 0 <= owner < len(players) and int(players[owner][2]) == 1:
        return 1 + int(players[owner][4])
    return 1


def blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> Set[Tuple[int, int]]:
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
    blasts = []
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


def explosion_time_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
                         horizon: float = EXPLOSION_TIME_HORIZON) -> np.ndarray:
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


def danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
                 timer_threshold: int = 1) -> np.ndarray:
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger

    plane = explosion_time_plane(grid, players, bombs)
    threshold = float(timer_threshold) / float(EXPLOSION_TIME_HORIZON) if EXPLOSION_TIME_HORIZON > 0 else 0.0
    danger[plane <= threshold] = 1.0
    return danger


def bfs_distance_to_targets(grid: np.ndarray, start: Tuple[int, int], targets: set,
                            bombs: np.ndarray, max_depth: int = 64) -> Optional[int]:
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
        for a in (LEFT, RIGHT, UP, DOWN):
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


def bfs_reachable_count(grid: np.ndarray, start: Tuple[int, int], bombs: np.ndarray,
                        max_depth: int = 3) -> int:
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
        for a in (LEFT, RIGHT, UP, DOWN):
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


def bfs_escape_available(grid: np.ndarray, start: Tuple[int, int], players: np.ndarray,
                         bombs: np.ndarray, max_depth: int = 6) -> int:
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
        for a in (LEFT, RIGHT, UP, DOWN):
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


def legal_actions(grid: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int],
                  bombs_left: int) -> List[int]:
    moves = [STOP]
    blocked = bomb_positions_set(bombs)
    for a in (LEFT, RIGHT, UP, DOWN):
        nr, nc = next_pos(my_pos, a)
        if passable(grid, nr, nc) and (nr, nc) not in blocked:
            moves.append(a)
    if bombs_left > 0 and my_pos not in blocked:
        moves.append(BOMB)
    return moves


def bfs_escape_action(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
                      start: Tuple[int, int]) -> int:
    """Tìm một hướng đi an toàn (không vào ô nguy hiểm ngay lập tức)."""
    blocked = bomb_positions_set(bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
    # thử các hướng đi trực tiếp
    q = deque()
    for a in (LEFT, RIGHT, UP, DOWN):
        nr, nc = next_pos(start, a)
        if (passable(grid, nr, nc) and (nr, nc) not in blocked
                and danger_now[nr, nc] == 0.0):
            q.append((nr, nc, a, 1))
    if not q and danger_now[start[0], start[1]] == 0.0:
        return STOP   # đứng yên an toàn

    seen = {start}
    while q:
        r, c, first, dist = q.popleft()
        if dist > 0 and danger_now[r, c] == 0.0:
            return first
        if dist >= 6:
            continue
        for a in (LEFT, RIGHT, UP, DOWN):
            nr, nc = next_pos((r, c), a)
            if (nr, nc) in seen:
                continue
            if not passable(grid, nr, nc):
                continue
            if (nr, nc) in blocked:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, first, dist + 1))
    return STOP


# -----------------------------------------------------------------------------
# Observation encoding (identical to training)
# -----------------------------------------------------------------------------
def encode_obs(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray,
               my_id: int, step: int) -> torch.Tensor:
    state = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # 0-4: static map
    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    # 5-8: player positions
    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                state[5 + pid, r, c] = 1.0

    # 9: chain-reaction explosion time
    state[9] = explosion_time_plane(grid, players, bombs)

    # my info
    me_alive = 0
    my_pos = (0, 0)
    bombs_left = 0
    bomb_radius = 1
    if my_id < len(players) and int(players[my_id][2]) == 1:
        me_alive = 1
        mr, mc = int(players[my_id][0]), int(players[my_id][1])
        my_pos = (mr, mc)
        if in_bounds(mr, mc):
            state[10, mr, mc] = 1.0
        bombs_left = int(players[my_id][3])
        bomb_radius = 1 + int(players[my_id][4])

    # 11: bombs_left
    state[11].fill(normalize_scalar(bombs_left, 5.0))

    # 12 & 13: bomb timer & radius
    if bombs is not None and len(bombs) > 0:
        for b in bombs:
            r, c, timer = int(b[0]), int(b[1]), int(b[2])
            if in_bounds(r, c):
                state[12, r, c] = 1.0 / float(max(timer, 1))
                owner = int(b[3]) if len(b) > 3 else -1
                state[13, r, c] = normalize_scalar(bomb_radius_for_owner(players, owner), 6.0)
    state[13][state[13] == 0] = normalize_scalar(bomb_radius, 6.0)

    if me_alive:
        item_pos = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
        enemy_pos = {
            (int(players[i][0]), int(players[i][1]))
            for i in range(4)
            if i != my_id and i < len(players) and int(players[i][2]) == 1
        }
        d_item = bfs_distance_to_targets(grid, my_pos, item_pos, bombs)
        d_enemy = bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs)

        state[14].fill(norm_dist(d_item))
        state[15].fill(norm_dist(d_enemy))
        state[16].fill(normalize_scalar(bfs_reachable_count(grid, my_pos, bombs, max_depth=3), 20.0))
        state[17].fill(float(bfs_escape_available(grid, my_pos, players, bombs, max_depth=6)))

        # box density
        r0, c0 = my_pos
        box_cnt = 0
        total_cnt = 0
        for rr in range(max(0, r0 - 2), min(BOARD_SIZE, r0 + 3)):
            for cc in range(max(0, c0 - 2), min(BOARD_SIZE, c0 + 3)):
                total_cnt += 1
                if int(grid[rr, cc]) == 2:
                    box_cnt += 1
        state[18].fill(normalize_scalar(box_cnt, float(max(total_cnt, 1))))

        # enemy pressure
        if d_enemy is None:
            state[19].fill(0.0)
        else:
            state[19].fill(1.0 - norm_dist(d_enemy))
    else:
        state[14].fill(1.0)
        state[15].fill(1.0)
        state[16].fill(0.0)
        state[17].fill(0.0)
        state[18].fill(0.0)
        state[19].fill(0.0)

    state[20].fill(normalize_scalar(len(bombs) if bombs is not None else 0, 10.0))
    state[21].fill(float(me_alive))
    state[22].fill(normalize_scalar(step, float(MAX_STEPS)))
    state[23].fill(1.0 - normalize_scalar(step, float(MAX_STEPS)))

    return torch.from_numpy(state)


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

        # Load trained model
        self.model = BomberNet(input_channels=INPUT_CHANNELS, num_actions=6, width=64)
        model_path = os.path.join(os.path.dirname(__file__), "weights.pth")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Warm-up
        dummy = torch.zeros(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            _ = self.model(dummy)

    def act(self, obs: dict) -> int:
        if obs["players"][self.agent_id][2] == 0:
            return STOP

        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        my_id   = self.agent_id

        # Encode observation
        state_tensor = encode_obs(grid, players, bombs, my_id, self.step_count).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits = self.model(state_tensor)[0]   # (6,)

        # Legal actions
        my_x, my_y = int(players[my_id][0]), int(players[my_id][1])
        bombs_left = int(players[my_id][3])
        legal = legal_actions(grid, bombs, (my_x, my_y), bombs_left)

        # Mask illegal actions
        mask = torch.full((6,), -float('inf'))
        mask[legal] = 0.0
        masked_logits = logits + mask
        chosen_action = int(torch.argmax(masked_logits).item())

        # Safety net: if chosen move leads into immediate danger, find escape
        danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
        if chosen_action != BOMB and chosen_action != STOP:
            nx, ny = next_pos((my_x, my_y), chosen_action)
            if danger_now[nx, ny] > 0.0:
                escape = bfs_escape_action(grid, players, bombs, (my_x, my_y))
                if escape is not None:
                    chosen_action = escape

        self.step_count += 1
        return chosen_action