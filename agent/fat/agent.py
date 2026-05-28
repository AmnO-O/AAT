import os
import random
from collections import deque
from typing import Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


class Agent:
    """
    Inference-only agent for the updated 27-channel Bomberland model.

    Expected weight files next to this file:
      - model_bc_best.pth
      - model_bc.pth
    """

    BOARD_SIZE = 13
    INPUT_CHANNELS = 27
    NUM_ACTIONS = 6
    MAX_STEPS = 500
    EXPLOSION_TIME_HORIZON = 8.0

    # Contest mapping:
    # 0 STOP, 1 LEFT, 2 RIGHT, 3 UP, 4 DOWN, 5 PLACE_BOMB
    MOVES = {
        0: (0, 0),
        1: (0, -1),
        2: (0, 1),
        3: (-1, 0),
        4: (1, 0),
    }

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = BomberNet(self.INPUT_CHANNELS, self.NUM_ACTIONS)
        self.model.eval()
        self._load_weights()

    def _load_weights(self) -> None:
        candidates = [
            "model_bc_best.pth",
            "model_bc.pth",
            "model.pth",
            "weights.pth",
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state and len(state) == 1:
                    state = state["state_dict"]
                if isinstance(state, dict):
                    self.model.load_state_dict(state, strict=True)
                    return
            except Exception:
                continue

    def act(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        my_r = int(players[self.agent_id][0])
        my_c = int(players[self.agent_id][1])
        my_pos = (my_r, my_c)
        bombs_left = int(players[self.agent_id][3])

        # Use the trained policy first.
        with torch.no_grad():
            state = encode_obs(grid, players, bombs, self.agent_id, 0).unsqueeze(0)
            logits = self.model(state)
            action = int(torch.argmax(logits, dim=1).item())

        # Very small safety patch: never place a bomb when escape is obviously impossible.
        if action == 5 and bombs_left > 0:
            if not self._can_escape_after_placing(grid, players, bombs, my_pos):
                safe_moves = self._safe_moves(grid, players, bombs, my_pos)
                if safe_moves:
                    return int(random.choice(safe_moves))
                return 0

        # If current tile is in imminent danger, prefer a safe move.
        danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
        danger_soon = future_danger_plane(grid, players, bombs)
        if danger_now[my_r, my_c] > 0.0 or danger_soon[my_r, my_c] > 0.0:
            safe_moves = self._safe_moves(grid, players, bombs, my_pos)
            if safe_moves:
                # Prefer the move that improves escape margin the most.
                best_action = None
                best_score = -10**9
                for a in safe_moves:
                    npos = self._next_pos(my_pos, a)
                    score = escape_margin_from_position(grid, players, bombs, npos, max_depth=6)
                    score += self._open_neighbors(grid, npos, self._bomb_positions_set(bombs))
                    if score > best_score:
                        best_score = score
                        best_action = a
                if best_action is not None:
                    return int(best_action)

        return action

    def _next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dr, dc = self.MOVES[int(action)]
        return pos[0] + dr, pos[1] + dc

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def _passable(self, grid: np.ndarray, r: int, c: int) -> bool:
        return self._in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)

    def _bomb_positions_set(self, bombs: np.ndarray) -> Set[Tuple[int, int]]:
        if bombs is None or len(bombs) == 0:
            return set()
        return {(int(b[0]), int(b[1])) for b in bombs}

    def _safe_moves(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int]) -> list[int]:
        blocked = self._bomb_positions_set(bombs)
        danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
        danger_soon = future_danger_plane(grid, players, bombs)
        moves = []
        for a in (1, 2, 3, 4):
            nr, nc = self._next_pos(my_pos, a)
            if not self._passable(grid, nr, nc):
                continue
            if (nr, nc) in blocked:
                continue
            if danger_now[nr, nc] > 0.0:
                continue
            if danger_soon[nr, nc] > 0.0:
                continue
            moves.append(a)
        return moves

    def _can_escape_after_placing(
        self,
        grid: np.ndarray,
        players: np.ndarray,
        bombs: np.ndarray,
        my_pos: Tuple[int, int],
    ) -> bool:
        blocked = self._bomb_positions_set(bombs)
        blocked.discard(my_pos)

        extra_bomb = np.array([[my_pos[0], my_pos[1], 7, self.agent_id]], dtype=np.int8)
        if bombs is None or len(bombs) == 0:
            merged = extra_bomb
        else:
            merged = np.concatenate([bombs, extra_bomb], axis=0)

        return self._can_reach_safe_tile(
            grid=grid,
            start=my_pos,
            players=players,
            bombs=merged,
            blocked=blocked,
            start_time=0,
            max_depth=16,
        )

    def _can_reach_safe_tile(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        players: np.ndarray,
        bombs: np.ndarray,
        blocked: Set[Tuple[int, int]],
        start_time: int = 0,
        max_depth: int = 16,
    ) -> bool:
        deadlines = tile_earliest_explosion_times(grid, players, bombs)
        q = deque([(start, 0)])
        seen = {start}

        while q:
            pos, dist = q.popleft()
            cur_time = start_time + dist
            if dist > 0 and cur_time < int(deadlines[pos[0], pos[1]]):
                return True

            if dist >= max_depth:
                continue

            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if npos in blocked:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                arrive = start_time + dist + 1
                if arrive >= int(deadlines[npos[0], npos[1]]):
                    continue
                seen.add(npos)
                q.append((npos, dist + 1))

        return False

    def _open_neighbors(self, grid: np.ndarray, pos: Tuple[int, int], blocked: Set[Tuple[int, int]]) -> int:
        cnt = 0
        for a in (1, 2, 3, 4):
            nr, nc = self._next_pos(pos, a)
            if self._passable(grid, nr, nc) and (nr, nc) not in blocked:
                cnt += 1
        return cnt


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
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
    def __init__(self, input_channels: int = 27, num_actions: int = 6, width: int = 64):
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


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 13 and 0 <= c < 13


def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)


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


def tile_earliest_explosion_times(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray) -> np.ndarray:
    plane = np.full((13, 13), 10**9, dtype=np.int32)
    if bombs is None or len(bombs) == 0:
        return plane

    times = bomb_effective_explosion_times(grid, players, bombs)
    for i in range(len(bombs)):
        owner = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
        radius = bomb_radius_for_owner(players, owner)
        t = int(max(0, times[i]))
        for r, c in blast_tiles(grid, int(bombs[i][0]), int(bombs[i][1]), radius):
            if t < plane[r, c]:
                plane[r, c] = t
    return plane


def danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, timer_threshold: int = 1) -> np.ndarray:
    danger = np.zeros((13, 13), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return danger
    earliest = tile_earliest_explosion_times(grid, players, bombs)
    danger[earliest <= timer_threshold] = 1.0
    return danger


def chain_danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, chain_horizon: int = 3) -> np.ndarray:
    plane = np.zeros((13, 13), dtype=np.float32)
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
    plane = np.zeros((13, 13), dtype=np.float32)
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


def escape_margin_from_position(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    start: Tuple[int, int],
    max_depth: int = 6,
) -> float:
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
            npos = (pos[0] + Agent.MOVES[a][0], pos[1] + Agent.MOVES[a][1])
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
            npos = (pos[0] + Agent.MOVES[a][0], pos[1] + Agent.MOVES[a][1])
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
            npos = (pos[0] + Agent.MOVES[a][0], pos[1] + Agent.MOVES[a][1])
            if npos in seen:
                continue
            if npos in blocked:
                continue
            if not passable(grid, npos[0], npos[1]):
                continue
            seen.add(npos)
            q.append((npos, dist + 1))
    return count


def time_safe_escape_score(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> float:
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return 0.0
    my_pos = (int(players[my_id][0]), int(players[my_id][1]))
    margin = escape_margin_from_position(grid, players, bombs, my_pos, max_depth=6)
    if margin <= 0:
        return 0.0
    return float(np.clip(margin / 6.0, 0.0, 1.0))


def safe_to_bomb_plane(
    grid: np.ndarray,
    players: np.ndarray,
    bombs: np.ndarray,
    my_id: int,
) -> np.ndarray:
    plane = np.zeros((13, 13), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    if not in_bounds(my_r, my_c):
        return plane

    blocked = bomb_positions_set(bombs)
    if (my_r, my_c) in blocked:
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

    safe_exit = False
    for a in (1, 2, 3, 4):
        nr, nc = my_r + Agent.MOVES[a][0], my_c + Agent.MOVES[a][1]
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


def bottleneck_risk_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    plane = np.zeros((13, 13), dtype=np.float32)
    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    my_r, my_c = int(players[my_id][0]), int(players[my_id][1])
    blocked = bomb_positions_set(bombs)
    danger_now = danger_plane(grid, players, bombs, timer_threshold=1)
    explosion_times = tile_earliest_explosion_times(grid, players, bombs)

    for r in range(13):
        for c in range(13):
            if not passable(grid, r, c):
                continue
            if (r, c) in blocked:
                continue

            exits = 0
            fragile = 0
            for a in (1, 2, 3, 4):
                nr, nc = r + Agent.MOVES[a][0], c + Agent.MOVES[a][1]
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

            manhattan = abs(r - my_r) + abs(c - my_c)
            if manhattan <= 1:
                score = max(score, 0.75)
            elif manhattan <= 2:
                score = max(score, 0.45)

            plane[r, c] = score
    return plane


def normalize_scalar(x: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(np.clip(x / denom, 0.0, 1.0))


def norm_dist(d: Optional[int], cap: float = 24.0) -> float:
    if d is None:
        return 1.0
    return float(min(d, cap)) / cap


def encode_obs(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int) -> torch.Tensor:
    state = np.zeros((27, 13, 13), dtype=np.float32)

    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                state[5 + pid, r, c] = 1.0

    state[9] = tile_earliest_explosion_times(grid, players, bombs).astype(np.float32)
    state[10] = danger_plane(grid, players, bombs, timer_threshold=1)
    state[11] = chain_danger_plane(grid, players, bombs)
    state[12] = future_danger_plane(grid, players, bombs)

    my_pos = (0, 0)
    bombs_left = 0
    if my_id < len(players) and int(players[my_id][2]) == 1:
        mr, mc = int(players[my_id][0]), int(players[my_id][1])
        my_pos = (mr, mc)
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

    if my_id < len(players) and int(players[my_id][2]) == 1:
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
    state[23].fill(normalize_scalar(step, float(Agent.MAX_STEPS)))
    state[24] = enemy_current_bomb_pressure(grid, players, bombs, my_id)
    state[25] = enemy_future_bomb_pressure(grid, players, bombs, my_id)
    state[26] = bottleneck_risk_plane(grid, players, bombs, my_id)

    return torch.from_numpy(state)


def enemy_current_bomb_pressure(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    plane = np.zeros((13, 13), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane

    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    enemy_tiles = set()
    for i in range(4):
        if i == my_id or i >= len(players) or int(players[i][2]) != 1:
            continue
        enemy_tiles.add((int(players[i][0]), int(players[i][1])))

    for b in bombs:
        owner = int(b[3]) if len(b) > 3 else -1
        if owner == my_id:
            continue
        r, c = int(b[0]), int(b[1])
        radius = bomb_radius_for_owner(players, owner)
        for x, y in blast_tiles(grid, r, c, radius):
            if (x, y) in enemy_tiles:
                plane[x, y] = 1.0
    return plane


def enemy_future_bomb_pressure(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int) -> np.ndarray:
    plane = np.zeros((13, 13), dtype=np.float32)
    if bombs is None or len(bombs) == 0:
        return plane

    if my_id >= len(players) or int(players[my_id][2]) != 1:
        return plane

    effective = bomb_effective_explosion_times(grid, players, bombs)
    for i, b in enumerate(bombs):
        owner = int(b[3]) if len(b) > 3 else -1
        if owner == my_id:
            continue
        r, c = int(b[0]), int(b[1])
        radius = bomb_radius_for_owner(players, owner)
        t = float(max(0, int(effective[i])))
        score = 1.0 - min(t, Agent.EXPLOSION_TIME_HORIZON) / Agent.EXPLOSION_TIME_HORIZON
        if score <= 0:
            continue
        for x, y in blast_tiles(grid, r, c, radius):
            plane[x, y] = max(plane[x, y], score)
    return plane
