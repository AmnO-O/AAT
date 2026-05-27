import os
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

BOARD_SIZE = 13
INPUT_CHANNELS = 20
NUM_ACTIONS = 6
MAX_STEPS = 500

MOVES = {
    0: (0, 0),    # STOP
    1: (-1, 0),   # LEFT
    2: (1, 0),    # RIGHT
    3: (0, -1),   # UP
    4: (0, 1),    # DOWN
}


# -----------------------------------------------------------------------------
# Board helpers
# -----------------------------------------------------------------------------

def next_pos(pos: Tuple[int, int], action: int) -> Tuple[int, int]:
    dr, dc = MOVES[int(action)]
    return pos[0] + dr, pos[1] + dc


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def passable(grid: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)


def bomb_positions_set(bombs: np.ndarray) -> set:
    return {(int(b[0]), int(b[1])) for b in bombs}


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


def danger_plane(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, timer_threshold: int = 1) -> np.ndarray:
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        owner = int(b[3]) if len(b) > 3 else -1
        radius = 1
        if 0 <= owner < len(players) and int(players[owner][2]) == 1:
            radius = 1 + int(players[owner][4])
        if timer <= timer_threshold:
            for r, c in blast_tiles(grid, bx, by, radius):
                danger[r, c] = 1.0
    return danger


def bfs_distance_to_targets(
    grid: np.ndarray,
    start: Tuple[int, int],
    targets: set,
    bombs: np.ndarray,
    max_depth: int = 64,
) -> int | None:
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
        for a in [1, 2, 3, 4]:
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
        for a in [1, 2, 3, 4]:
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
        for a in [1, 2, 3, 4]:
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


def norm_dist(d: int | None, cap: float = 24.0) -> float:
    if d is None:
        return 1.0
    return float(min(d, cap)) / cap


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

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
        out = F.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out)
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
# Inference agent
# -----------------------------------------------------------------------------

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0
        self.device = torch.device("cpu")
        self.model = BomberNet().to(self.device)
        self.model.eval()
        self._load_checkpoint()

    def _load_checkpoint(self):
        candidates = [
            "model_bc.pth",
            "model_bc_best.pth",
            os.path.join(os.getcwd(), "model_bc.pth"),
            os.path.join(os.getcwd(), "weights.pth"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    state = torch.load(path, map_location=self.device)
                    self.model.load_state_dict(state, strict=True)
                    self.model.eval()
                    return
                except Exception:
                    continue
        self.model = None

    def act(self, obs: dict) -> int:
        try:
            grid = obs["map"]
            players = obs["players"]
            bombs = obs["bombs"]

            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                self.step += 1
                return 0

            my_r = int(players[self.agent_id][0])
            my_c = int(players[self.agent_id][1])
            my_pos = (my_r, my_c)
            bombs_left = int(players[self.agent_id][3])
            bomb_radius = 1 + int(players[self.agent_id][4])

            bomb_positions = bomb_positions_set(bombs)
            danger_now = danger_plane(grid, players, bombs, timer_threshold=1)

            # If we're in immediate danger, escape first.
            if danger_now[my_r, my_c] > 0.0:
                escape = self._escape_action(grid, players, bombs, my_pos)
                self.step += 1
                if escape is not None:
                    return int(escape)

            if self.model is None:
                action = self._rule_fallback(grid, players, bombs, my_pos, bombs_left, bomb_radius)
                self.step += 1
                return int(action)

            state = self._encode_obs(grid, players, bombs, self.agent_id, self.step).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(state).squeeze(0).cpu().numpy()

            legal = self._legal_actions(grid, bombs, my_pos, bombs_left)
            action = self._select_action_with_safety(
                logits=logits,
                legal_actions=legal,
                grid=grid,
                players=players,
                bombs=bombs,
                my_pos=my_pos,
                bombs_left=bombs_left,
                bomb_radius=bomb_radius,
                bomb_positions=bomb_positions,
            )

            self.step += 1
            return int(action)

        except Exception:
            self.step += 1
            return 0

    def _encode_obs(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int) -> torch.Tensor:
        """
        Matches the last training pipeline:
        0 wall, 1 box, 2 grass, 3 radius item, 4 capacity item,
        5-8 players,
        9 danger now,
        10 my position,
        11 bombs left,
        12 bomb timers,
        13 bomb radius,
        14 item distance,
        15 enemy distance,
        16 reachable count,
        17 escape available,
        18 step ratio,
        19 alive flag
        """
        C, H, W = INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
        state = np.zeros((C, H, W), dtype=np.float32)

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

        state[9] = danger_plane(grid, players, bombs, timer_threshold=1)

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

        state[11] = float(bombs_left) / 5.0

        for b in bombs:
            r, c, timer = int(b[0]), int(b[1]), int(b[2])
            if in_bounds(r, c):
                state[12, r, c] = 1.0 / float(timer + 1)

        state[13] = float(bomb_radius) / 6.0

        if me_alive:
            item_pos = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
            enemy_pos = {
                (int(players[i][0]), int(players[i][1]))
                for i in range(4)
                if i != my_id and i < len(players) and int(players[i][2]) == 1
            }

            d_item = bfs_distance_to_targets(grid, my_pos, item_pos, bombs)
            d_enemy = bfs_distance_to_targets(grid, my_pos, enemy_pos, bombs)

            state[14] = norm_dist(d_item)
            state[15] = norm_dist(d_enemy)
            state[16] = float(bfs_reachable_count(grid, my_pos, bombs, max_depth=3)) / 20.0
            state[17] = float(bfs_escape_available(grid, my_pos, players, bombs, max_depth=6))
        else:
            state[14] = 1.0
            state[15] = 1.0
            state[16] = 0.0
            state[17] = 0.0

        state[18] = float(step) / float(MAX_STEPS)
        state[19] = float(me_alive)

        return torch.from_numpy(state)

    def _legal_actions(self, grid: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int], bombs_left: int) -> List[int]:
        legal = [0]
        bomb_positions = bomb_positions_set(bombs)

        for a in [1, 2, 3, 4]:
            nr, nc = next_pos(my_pos, a)
            if passable(grid, nr, nc) and (nr, nc) not in bomb_positions:
                legal.append(a)

        if bombs_left > 0 and my_pos not in bomb_positions:
            legal.append(5)

        return legal

    def _move_to_nearest_safe(self, grid: np.ndarray, start: Tuple[int, int], players: np.ndarray, bombs: np.ndarray, search_depth: int = 8):
        blocked = bomb_positions_set(bombs)
        danger = danger_plane(grid, players, bombs, timer_threshold=1)
        q = deque([(start, 0, None)])
        seen = {start}

        while q:
            pos, dist, first_action = q.popleft()
            if dist > 0 and danger[pos[0], pos[1]] == 0.0:
                return first_action
            if dist >= search_depth:
                continue

            for a in [1, 2, 3, 4]:
                npos = next_pos(pos, a)
                if npos in seen:
                    continue
                if npos in blocked:
                    continue
                if not passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, dist + 1, a if first_action is None else first_action))

        return None

    def _escape_action(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int]):
        action = self._move_to_nearest_safe(grid, my_pos, players, bombs, search_depth=8)
        if action is not None:
            return int(action)

        # fallback to any move that avoids immediate danger if possible
        danger = danger_plane(grid, players, bombs, timer_threshold=1)
        bomb_positions = bomb_positions_set(bombs)
        safe_moves = []
        for a in [1, 2, 3, 4]:
            npos = next_pos(my_pos, a)
            if passable(grid, npos[0], npos[1]) and npos not in bomb_positions and danger[npos[0], npos[1]] == 0.0:
                safe_moves.append(a)
        if safe_moves:
            return int(random.choice(safe_moves))
        return 0

    def _count_boxes_in_blast(self, grid: np.ndarray, pos: Tuple[int, int], radius: int) -> int:
        return sum(1 for r, c in blast_tiles(grid, pos[0], pos[1], radius) if int(grid[r, c]) == 2)

    def _can_hit_enemy_with_bomb(self, grid: np.ndarray, pos: Tuple[int, int], players: np.ndarray, radius: int) -> bool:
        mx, my = pos
        for pid in range(4):
            if pid >= len(players) or int(players[pid][2]) != 1:
                continue
            ex, ey = int(players[pid][0]), int(players[pid][1])
            if mx == ex and abs(ey - my) <= radius:
                step = 1 if ey > my else -1
                clear = True
                for y in range(my + step, ey, step):
                    if int(grid[mx, y]) in (1, 2):
                        clear = False
                        break
                if clear:
                    return True
            if my == ey and abs(ex - mx) <= radius:
                step = 1 if ex > mx else -1
                clear = True
                for x in range(mx + step, ex, step):
                    if int(grid[x, my]) in (1, 2):
                        clear = False
                        break
                if clear:
                    return True
        return False

    def _can_escape_after_placing(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, pos: Tuple[int, int], bomb_radius: int) -> bool:
        my_blast = blast_tiles(grid, pos[0], pos[1], bomb_radius)
        combined_danger = set(my_blast)
        combined_danger |= {tuple(map(int, b[:2])) for b in bombs if len(b) >= 2}

        # quick local BFS to any safe tile
        blocked = bomb_positions_set(bombs)
        q = deque([(pos, 0)])
        seen = {pos}
        while q:
            cur, dist = q.popleft()
            if dist > 0 and cur not in combined_danger:
                return True
            if dist >= 6:
                continue
            for a in [1, 2, 3, 4]:
                npos = next_pos(cur, a)
                if npos in seen:
                    continue
                if npos in blocked:
                    continue
                if not passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, dist + 1))
        return False

    def _rule_fallback(self, grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_pos: Tuple[int, int], bombs_left: int, bomb_radius: int) -> int:
        """
        Simple safe fallback: escape > items > boxes > enemy pressure > random safe move.
        """
        danger = danger_plane(grid, players, bombs, timer_threshold=1)
        if danger[my_pos[0], my_pos[1]] > 0.0:
            esc = self._escape_action(grid, players, bombs, my_pos)
            if esc is not None:
                return int(esc)

        bomb_positions = bomb_positions_set(bombs)
        legal_moves = []
        for a in [1, 2, 3, 4]:
            npos = next_pos(my_pos, a)
            if passable(grid, npos[0], npos[1]) and npos not in bomb_positions:
                legal_moves.append(a)

        items = {(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))}
        if items and legal_moves:
            best = min(legal_moves, key=lambda a: min(abs(next_pos(my_pos, a)[0] - r) + abs(next_pos(my_pos, a)[1] - c) for r, c in items))
            return int(best)

        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_hit = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            enemy_hit = self._can_hit_enemy_with_bomb(grid, my_pos, players, bomb_radius)
            if (boxes_hit > 0 or enemy_hit) and self._can_escape_after_placing(grid, players, bombs, my_pos, bomb_radius):
                return 5

        safe_moves = [a for a in legal_moves if danger[next_pos(my_pos, a)[0], next_pos(my_pos, a)[1]] == 0.0]
        if safe_moves:
            return int(random.choice(safe_moves))
        return 0


# Optional alias if your runner expects `class Agent` only.