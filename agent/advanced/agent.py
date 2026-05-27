"""
Bomberland Agent – Hybrid Rule + Neural MCTS
============================================
Matches the training architecture: BomberNet with ResidualBlock, width=128,
AdaptiveAvgPool2d, SiLU, Dropout2d, etc.
"""

import random
import time
import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants ─────────────────────────────────────────────────────────────
STOP    = 0
LEFT    = 1
RIGHT   = 2
UP      = 3
DOWN    = 4
BOMB    = 5

MOVES = {STOP: (0, 0), LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1)}
BOARD_SIZE = 13
INPUT_CHANNELS = 20
MAX_STEPS = 500


# ─── Network architecture (identical to training script) ────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.silu(out)
        return out


class BomberNet(nn.Module):
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_actions: int = 6, width: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, dropout=0.05, dilation=1),
            ResidualBlock(width, dropout=0.05, dilation=1),
            ResidualBlock(width, dropout=0.05, dilation=2),
            ResidualBlock(width, dropout=0.05, dilation=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.head(x)
        return x


# ─── Helper functions (taken from the training script) ─────────────────────
def _in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

def _passable(grid, r, c):
    return _in_bounds(r, c) and int(grid[r, c]) in (0, 3, 4)

def _blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1, radius+1):
            r, c = bx + dr*d, by + dc*d
            if not _in_bounds(r, c):
                break
            cell = int(grid[r, c])
            if cell == 1:
                break
            tiles.add((r, c))
            if cell == 2:
                break
    return tiles

def _danger_plane(grid, players, bombs, timer_threshold=1):
    danger = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        owner = int(b[3]) if len(b) > 3 else -1
        radius = 1
        if 0 <= owner < len(players) and int(players[owner][2]) == 1:
            radius = 1 + int(players[owner][4])
        if timer <= timer_threshold:
            for r, c in _blast_tiles(grid, bx, by, radius):
                danger[r, c] = 1.0
    return danger

def _reachable_count(grid, start, bombs, max_depth=3):
    blocked = {(int(b[0]), int(b[1])) for b in bombs}
    q = deque([(start, 0)])
    seen = {start}
    cnt = 0
    while q:
        pos, d = q.popleft()
        if d > 0:
            cnt += 1
        if d >= max_depth:
            continue
        for a in (LEFT, RIGHT, UP, DOWN):
            npos = (pos[0]+MOVES[a][0], pos[1]+MOVES[a][1])
            if npos in seen or npos in blocked or not _passable(grid, *npos):
                continue
            seen.add(npos)
            q.append((npos, d+1))
    return cnt

def _can_escape_feat(grid, start, players, bombs, max_depth=6):
    blocked = {(int(b[0]), int(b[1])) for b in bombs}
    danger = _danger_plane(grid, players, bombs, timer_threshold=1)
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if d > 0 and danger[pos[0], pos[1]] == 0.0:
            return 1
        if d >= max_depth:
            continue
        for a in (LEFT, RIGHT, UP, DOWN):
            npos = (pos[0]+MOVES[a][0], pos[1]+MOVES[a][1])
            if npos in seen or npos in blocked or not _passable(grid, *npos):
                continue
            seen.add(npos)
            q.append((npos, d+1))
    return 0

def encode_obs(grid, players, bombs, my_id, step):
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
            if _in_bounds(r, c):
                state[5+pid, r, c] = 1.0
    state[9] = _danger_plane(grid, players, bombs, timer_threshold=1)
    me = players[my_id]
    me_alive = int(me[2]) == 1
    my_pos = (0,0)
    bombs_left = 0
    bomb_radius = 1
    if me_alive:
        mr, mc = int(me[0]), int(me[1])
        my_pos = (mr, mc)
        if _in_bounds(mr, mc):
            state[10, mr, mc] = 1.0
        bombs_left = int(me[3])
        bomb_radius = 1 + int(me[4])
    state[11] = float(bombs_left) / 5.0
    for b in bombs:
        r, c, timer = int(b[0]), int(b[1]), int(b[2])
        if _in_bounds(r, c):
            state[12, r, c] = 1.0 / float(timer + 1)
    state[13] = float(bomb_radius) / 6.0
    item_pos = [(int(r), int(c)) for r, c in np.argwhere((grid == 3) | (grid == 4))]
    enemy_pos = [(int(players[i][0]), int(players[i][1])) for i in range(4)
                 if i != my_id and i < len(players) and int(players[i][2]) == 1]
    if me_alive:
        state[14] = min([abs(r-my_pos[0])+abs(c-my_pos[1]) for r,c in item_pos])/20.0 if item_pos else 1.0
        state[15] = min([abs(r-my_pos[0])+abs(c-my_pos[1]) for r,c in enemy_pos])/20.0 if enemy_pos else 1.0
        state[16] = float(_reachable_count(grid, my_pos, bombs, max_depth=3)) / 20.0
        state[17] = float(_can_escape_feat(grid, my_pos, players, bombs, max_depth=6))
    else:
        state[14] = 1.0
        state[15] = 1.0
        state[16] = 0.0
        state[17] = 0.0
    state[18] = float(step) / float(MAX_STEPS)
    alive = sum(1 for p in players if int(p[2]) == 1)
    state[19] = float(alive) / 4.0
    return torch.from_numpy(state)


# ─── Lightweight simulator for MCTS ────────────────────────────────────────
class SimState:
    def __init__(self, grid, players, bombs):
        self.grid    = grid.copy()
        self.players = players.copy()
        self.bombs   = bombs.copy() if len(bombs) > 0 else np.empty((0,4), dtype=np.int8)

    def copy(self):
        s = object.__new__(SimState)
        s.grid    = self.grid.copy()
        s.players = self.players.copy()
        s.bombs   = self.bombs.copy() if len(self.bombs) > 0 else np.empty((0,4), dtype=np.int8)
        return s

    def passable(self, x, y):
        return _in_bounds(x, y) and int(self.grid[x, y]) in (0, 3, 4)

    def blast_tiles(self, bx, by, radius):
        return _blast_tiles(self.grid, bx, by, radius)

    def danger_tiles(self):
        dnow, dsoon = set(), set()
        for b in self.bombs:
            bx, by, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            if timer <= 0:
                continue
            radius = max(1, int(self.players[owner][4])+1) if 0<=owner<len(self.players) else 2
            blast = self.blast_tiles(bx, by, radius)
            dsoon |= blast
            if timer <= 2:
                dnow |= blast
        return dnow, dsoon

    def get_legal_actions(self, pid):
        p = self.players[pid]
        if p[2] == 0:
            return [STOP]
        x, y = int(p[0]), int(p[1])
        bp = {(b[0], b[1]) for b in self.bombs}
        acts = [STOP]
        for a in (LEFT, RIGHT, UP, DOWN):
            nx, ny = x+MOVES[a][0], y+MOVES[a][1]
            if self.passable(nx, ny) and (nx, ny) not in bp:
                acts.append(a)
        if p[3] > 0 and (x, y) not in bp:
            acts.append(BOMB)
        return acts

    def step(self, actions):
        rewards = {aid:0.0 for aid in actions}
        bp = {(b[0], b[1]) for b in self.bombs}
        new_pos = {}
        for aid, a in actions.items():
            if self.players[aid][2] != 1:
                continue
            cx, cy = int(self.players[aid][0]), int(self.players[aid][1])
            if a in (1,2,3,4):
                dx, dy = MOVES[a]
                nx, ny = cx+dx, cy+dy
                new_pos[aid] = (nx, ny) if (self.passable(nx, ny) and (nx, ny) not in bp) else (cx, cy)
            else:
                new_pos[aid] = (cx, cy)
        for aid, (nx, ny) in new_pos.items():
            self.players[aid][0], self.players[aid][1] = nx, ny

        # items
        for aid in new_pos:
            if self.players[aid][2] != 1:
                continue
            x, y = self.players[aid][0], self.players[aid][1]
            cell = int(self.grid[x, y])
            if cell == 3:
                self.players[aid][4] = min(self.players[aid][4]+1, 4)
                self.grid[x, y] = 0
                rewards[aid] += 3.0
            elif cell == 4:
                self.players[aid][3] = min(self.players[aid][3]+1, 5)
                self.grid[x, y] = 0
                rewards[aid] += 3.0

        # bombs
        cur_bp = {(b[0], b[1]) for b in self.bombs}
        for aid, a in actions.items():
            if self.players[aid][2] != 1:
                continue
            if a == 5 and self.players[aid][3] > 0:
                x, y = int(self.players[aid][0]), int(self.players[aid][1])
                if (x, y) not in cur_bp:
                    self.bombs = np.vstack([self.bombs, np.array([[x, y, 7, aid]], dtype=np.int8)]) if len(self.bombs) > 0 else np.array([[x, y, 7, aid]], dtype=np.int8)
                    self.players[aid][3] -= 1
                    cur_bp.add((x, y))

        # timers
        for i in range(len(self.bombs)):
            self.bombs[i][2] -= 1

        # explosions
        exploded_mask = np.zeros(len(self.bombs), dtype=bool)
        while True:
            new_expl = False
            blast_tiles_all = set()
            for i, b in enumerate(self.bombs):
                if exploded_mask[i] or b[2] > 0:
                    continue
                bx, by = int(b[0]), int(b[1])
                owner = int(b[3])
                radius = max(1, int(self.players[owner][4])+1) if 0<=owner<len(self.players) else 2
                blast = self.blast_tiles(bx, by, radius)
                blast_tiles_all.update(blast)
                exploded_mask[i] = True
                new_expl = True
            if not new_expl:
                break
            for pid in actions:
                if self.players[pid][2] == 1 and (int(self.players[pid][0]), int(self.players[pid][1])) in blast_tiles_all:
                    self.players[pid][2] = 0
                    rewards[pid] -= 500.0
                    for oid in actions:
                        if oid != pid and self.players[oid][2] == 1:
                            rewards[oid] += 100.0
            for (x, y) in blast_tiles_all:
                if _in_bounds(x, y) and int(self.grid[x, y]) == 2:
                    self.grid[x, y] = 0
            # chain
            for b in self.bombs:
                if (b[0], b[1]) in blast_tiles_all:
                    b[2] = 0
            self.bombs = self.bombs[~exploded_mask]
            if len(self.bombs) == 0:
                self.bombs = np.empty((0,4), dtype=np.int8)
            exploded_mask = np.zeros(len(self.bombs), dtype=bool)

        # restore bomb capacity
        for b in self.bombs:
            owner = int(b[3])
            if 0<=owner<len(self.players):
                self.players[owner][3] = min(self.players[owner][3]+1, 5)

        return rewards

    def is_terminal(self):
        alive = sum(1 for p in self.players if p[2] == 1)
        return alive <= 1


# ─── MCTS Node ─────────────────────────────────────────────────────────────
class MCTSNode:
    def __init__(self, state, my_id, parent=None, action=None, prior=1.0):
        self.state = state
        self.my_id = my_id
        self.parent = parent
        self.action = action
        self.visits = 0
        self.total_value = 0.0
        self.children = {}
        self.untried_actions = state.get_legal_actions(my_id) if state is not None else []
        self.prior = prior

    def puct(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        q = self.total_value / self.visits
        u = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u


# ─── Main Agent ────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

        # Load trained model (same architecture as training)
        self.model = BomberNet(input_channels=INPUT_CHANNELS, num_actions=6, width=128)
        model_path = os.path.join(os.path.dirname(__file__), "weights.pth")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # warm-up to avoid first-act overhead
        dummy = torch.zeros(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        with torch.no_grad():
            _ = self.model(dummy)

    def act(self, obs):
        if obs["players"][self.agent_id][2] == 0:
            return STOP
        alive = sum(1 for p in obs["players"] if p[2] == 1)
        if alive > 2:
            return self._rule_act(obs)
        else:
            return self._mcts_act(obs)

    # ═══════════════════ Rule-based (early game) ═════════════════════════════
    def _rule_act(self, obs):
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        me      = players[self.agent_id]
        my_pos  = (int(me[0]), int(me[1]))
        bleft   = int(me[3])
        bradius = max(1, int(me[4])+1)
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs}
        enemies  = [(int(p[0]), int(p[1])) for i, p in enumerate(players)
                    if i != self.agent_id and p[2] == 1]
        blocked = (set(enemies) | bomb_pos) - {my_pos}

        dnow, dsoon = self._danger(grid, bombs, players)

        # 1. escape
        if my_pos in dnow:
            return self._escape_bfs(grid, my_pos, blocked, dnow, dsoon)
        if my_pos in dsoon:
            mv = self._bfs_to_safe(grid, my_pos, blocked, dsoon)
            if mv is not None:
                return mv

        # 2. collect items
        items = {(x, y) for x in range(13) for y in range(13) if grid[x, y] in (3, 4)}
        if items:
            mv = self._bfs(grid, my_pos, items, blocked, dsoon)
            if mv is not None:
                return mv

        # 3. bomb if hits enemy or boxes
        if bleft > 0 and my_pos not in bomb_pos:
            blast = _blast_tiles(grid, my_pos[0], my_pos[1], bradius)
            hit_enemy = any(e in blast for e in enemies)
            box_cnt = sum(1 for t in blast if grid[t[0], t[1]] == 2)
            if (hit_enemy or box_cnt >= 1) and self._can_escape(grid, my_pos, blocked, dsoon, bradius):
                return BOMB

        # 4. move toward box-farming spots
        bspots = self._box_spots(grid, blocked)
        if bspots:
            mv = self._bfs(grid, my_pos, bspots, blocked, dsoon)
            if mv is not None:
                return mv

        # 5. chase enemy
        if enemies:
            mv = self._bfs(grid, my_pos, set(enemies), blocked, dsoon)
            if mv is not None:
                return mv

        # 6. random safe move
        safe = [a for a in [LEFT,RIGHT,UP,DOWN] if (
            _passable(grid, my_pos[0]+MOVES[a][0], my_pos[1]+MOVES[a][1])
            and (my_pos[0]+MOVES[a][0], my_pos[1]+MOVES[a][1]) not in blocked
            and (my_pos[0]+MOVES[a][0], my_pos[1]+MOVES[a][1]) not in dsoon)]
        return random.choice(safe) if safe else STOP

    # ─── MCTS for 1v1 ──────────────────────────────────────────────────────
    def _mcts_act(self, obs, budget=0.085):
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        my_id   = self.agent_id
        alive_ids = [i for i, p in enumerate(players) if p[2] == 1]
        if len(alive_ids) <= 1:
            return self._rule_act(obs)
        enemy_id = next(i for i in alive_ids if i != my_id)

        init_state = SimState(
            grid.astype(np.int8),
            np.array([[int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])] for p in players]),
            bombs.astype(np.int8) if len(bombs) > 0 else np.empty((0,4), dtype=np.int8)
        )

        root = MCTSNode(init_state, my_id)
        root.visits = 1

        # compute prior from network for root
        state_tensor = encode_obs(grid, players, bombs, my_id, self.step_count).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0).cpu().numpy()
        legal = root.untried_actions   # same as init_state.get_legal_actions(my_id)
        probs = self._softmax_over_legal(logits, legal)

        # pre-expand children
        for a in legal:
            s = init_state.copy()
            enemy_act = self._enemy_policy(s, enemy_id, my_id)
            s.step({my_id: a, enemy_id: enemy_act})
            child = MCTSNode(s, my_id, parent=root, action=a, prior=probs[a])
            root.children[a] = child

        deadline = time.time() + budget
        while time.time() < deadline:
            # selection
            node = root
            while node.children and not node.state.is_terminal():
                best_a = max(node.children, key=lambda a: node.children[a].puct())
                node = node.children[best_a]

            # expansion
            if not node.state.is_terminal() and node.untried_actions:
                action = node.untried_actions.pop(0)
                s = node.state.copy()
                enemy_act = self._enemy_policy(s, enemy_id, my_id)
                s.step({my_id: action, enemy_id: enemy_act})
                # compute prior for this new child
                ns_tensor = encode_obs(s.grid, s.players, s.bombs, my_id, 0).unsqueeze(0)
                with torch.no_grad():
                    new_logits = self.model(ns_tensor).squeeze(0).cpu().numpy()
                new_legal = s.get_legal_actions(my_id)
                new_probs = self._softmax_over_legal(new_logits, new_legal)
                child = MCTSNode(s, my_id, parent=node, action=action, prior=new_probs[action])
                node.children[action] = child
                node = child

            # rollout from node
            value = self._rollout(node.state.copy(), my_id, enemy_id, depth=20)

            # backprop
            while node is not None:
                node.visits += 1
                node.total_value += value
                value = -value   # zero-sum
                node = node.parent

        if not root.children:
            return self._rule_act(obs)
        best_a = max(root.children, key=lambda a: root.children[a].visits)
        self.step_count += 1
        return best_a

    def _enemy_policy(self, state, eid, mid):
        legal = state.get_legal_actions(eid)
        if not legal:
            return STOP
        tensor = encode_obs(state.grid, state.players, state.bombs, eid, 0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0).cpu().numpy()
        mask = np.ones(6) * (-1e9)
        mask[list(legal)] = 0.0
        return int(np.argmax(logits + mask))

    def _rollout(self, state, my_id, enemy_id, depth):
        s = state.copy()
        disc = 1.0
        total = 0.0
        for _ in range(depth):
            if s.is_terminal() or s.players[my_id][2] != 1 or s.players[enemy_id][2] != 1:
                break
            my_act = self._heuristic_rollout_act(s, my_id)
            en_act = self._heuristic_rollout_act(s, enemy_id)
            rewards = s.step({my_id: my_act, enemy_id: en_act})
            total += disc * rewards.get(my_id, 0.0)
            disc *= 0.95
        my_alive = s.players[my_id][2] == 1
        en_alive = s.players[enemy_id][2] == 1
        if my_alive and not en_alive:
            total += disc * 600.0
        elif not my_alive and en_alive:
            total -= disc * 400.0
        return total

    def _heuristic_rollout_act(self, state, pid):
        p = state.players[pid]
        if p[2] != 1:
            return STOP
        pos = (int(p[0]), int(p[1]))
        bp = {(b[0], b[1]) for b in state.bombs}
        dnow, dsoon = state.danger_tiles()
        if pos in dnow or pos in dsoon:
            for a in random.sample([LEFT,RIGHT,UP,DOWN], 4):
                nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a
            return STOP
        # grab nearby item
        for a in [LEFT,RIGHT,UP,DOWN]:
            nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
            if state.passable(nx, ny) and (nx, ny) not in dsoon and int(state.grid[nx, ny]) in (3,4):
                return a
        safe = [a for a in [LEFT,RIGHT,UP,DOWN,STOP] if a==STOP or (
            state.passable(pos[0]+MOVES[a][0], pos[1]+MOVES[a][1])
            and (pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]) not in dsoon)]
        return random.choice(safe) if safe else STOP

    # ─── Shared navigation / utility ────────────────────────────────────────
    def _danger(self, grid, bombs, players):
        dnow, dsoon = set(), set()
        for b in bombs:
            bx, by, t = int(b[0]), int(b[1]), int(b[2])
            oid = int(b[3]) if len(b) > 3 else -1
            if t <= 0:
                continue
            r = max(1, int(players[oid][4])+1) if 0<=oid<len(players) else 2
            blast = _blast_tiles(grid, bx, by, r)
            dsoon |= blast
            if t <= 2:
                dnow |= blast
        return dnow, dsoon

    def _escape_bfs(self, grid, pos, blocked, dnow, dsoon):
        q, seen = deque([(pos, None)]), {pos}
        while q:
            p, first = q.popleft()
            if p not in dsoon and first is not None:
                return first
            for a in [LEFT,RIGHT,UP,DOWN]:
                nx, ny = p[0]+MOVES[a][0], p[1]+MOVES[a][1]
                npos = (nx, ny)
                if npos in seen or not _passable(grid, nx, ny) or npos in blocked or npos in dnow:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        for a in [LEFT,RIGHT,UP,DOWN]:
            nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
            if _passable(grid, nx, ny) and (nx, ny) not in blocked and (nx, ny) not in dnow:
                return a
        return STOP

    def _bfs_to_safe(self, grid, pos, blocked, dsoon):
        q, seen = deque([(pos, None)]), {pos}
        while q:
            p, first = q.popleft()
            if p not in dsoon and first is not None:
                return first
            for a in [LEFT,RIGHT,UP,DOWN]:
                nx, ny = p[0]+MOVES[a][0], p[1]+MOVES[a][1]
                npos = (nx, ny)
                if npos in seen or not _passable(grid, nx, ny) or npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        return None

    def _bfs(self, grid, start, targets, blocked, avoid):
        if not targets:
            return None
        q, seen = deque([(start, None)]), {start}
        while q:
            pos, first = q.popleft()
            if pos in targets and first is not None:
                return first
            for a in [LEFT,RIGHT,UP,DOWN]:
                nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
                npos = (nx, ny)
                if npos in seen or not _passable(grid, nx, ny):
                    continue
                if npos in blocked and npos not in targets:
                    continue
                if npos in avoid:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        return None

    def _can_escape(self, grid, pos, blocked, dsoon, radius):
        blast = _blast_tiles(grid, pos[0], pos[1], radius)
        combined = dsoon | blast
        q, seen = deque([(pos, 0)]), {pos}
        while q:
            p, d = q.popleft()
            if p not in combined and d > 0:
                return True
            if d >= 8:
                continue
            for a in [LEFT,RIGHT,UP,DOWN]:
                nx, ny = p[0]+MOVES[a][0], p[1]+MOVES[a][1]
                npos = (nx, ny)
                if npos in seen or not _passable(grid, nx, ny) or npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, d+1))
        return False

    def _box_spots(self, grid, blocked):
        spots = set()
        for x in range(13):
            for y in range(13):
                if grid[x, y] != 2:
                    continue
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if _passable(grid, nx, ny) and (nx, ny) not in blocked:
                        spots.add((nx, ny))
        return spots

    def _softmax_over_legal(self, logits, legal):
        exp = np.exp(logits[list(legal)] - np.max(logits[list(legal)]))
        sum_exp = exp.sum()
        probs = np.zeros(6)
        for idx, a in enumerate(legal):
            probs[a] = exp[idx] / sum_exp if sum_exp > 0 else 1.0/len(legal)
        return probs