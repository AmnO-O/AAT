"""
Bomberland Agent – Neural + MCTS
================================
Sử dụng mô hình CNN đã huấn luyện (BC+DAgger) để:
  - Giai đoạn >2 người: chọn hành động tốt nhất từ policy (argmax sau khi che các nước không hợp lệ).
  - Giai đoạn 1v1: MCTS với prior từ policy, rollout ngẫu nhiên có heuristic.

Yêu cầu: file model_bc_best.pth đặt cùng thư mục.
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


# ════════════════════════════════════════════════════════════════════════════
#  Hằng số
# ════════════════════════════════════════════════════════════════════════════
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

# ─────────────────────────────────────────────────────────────────────────────
#  Định nghĩa kiến trúc mạng (giống hệt lúc huấn luyện)
# ─────────────────────────────────────────────────────────────────────────────
class BomberNet(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),       # suy luận thì tắt dropout
            nn.Linear(256, 6),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Hàm mã hoá trạng thái (đồng bộ với script huấn luyện)
# ─────────────────────────────────────────────────────────────────────────────
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

    # 0‑4: map cơ bản
    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    # 5‑8: vị trí từng người chơi
    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if _in_bounds(r, c):
                state[5+pid, r, c] = 1.0

    # 9: nguy hiểm tức thì
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


# ─────────────────────────────────────────────────────────────────────────────
#  SimState – mô phỏng nhẹ cho MCTS
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self, grid, players, bombs):
        self.grid    = grid.copy()
        self.players = players.copy()
        self.bombs   = bombs.copy() if len(bombs)>0 else np.empty((0,4), dtype=np.int8)

    def copy(self):
        s = object.__new__(SimState)
        s.grid    = self.grid.copy()
        s.players = self.players.copy()
        s.bombs   = self.bombs.copy() if len(self.bombs)>0 else np.empty((0,4), dtype=np.int8)
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

    def step(self, actions):
        rewards = {aid:0.0 for aid in actions}
        bp = {(b[0],b[1]) for b in self.bombs}
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
        cur_bp = {(b[0],b[1]) for b in self.bombs}
        for aid, a in actions.items():
            if self.players[aid][2] != 1:
                continue
            if a == 5 and self.players[aid][3] > 0:
                x, y = int(self.players[aid][0]), int(self.players[aid][1])
                if (x, y) not in cur_bp:
                    self.bombs = np.vstack([self.bombs, np.array([[x, y, 7, aid]], dtype=np.int8)]) if len(self.bombs)>0 else np.array([[x, y, 7, aid]], dtype=np.int8)
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
        alive = sum(1 for p in self.players if p[2]==1)
        return alive <= 1


# ─────────────────────────────────────────────────────────────────────────────
#  MCTS node
# ─────────────────────────────────────────────────────────────────────────────
class _MCTSNode:
    __slots__ = ('visits','total_value','children','untried','parent','action','prior')
    def __init__(self, untried, prior=None, parent=None, action=None):
        self.visits = 0
        self.total_value = 0.0
        self.children = {}
        self.untried = list(untried)
        random.shuffle(self.untried)
        self.prior = prior if prior else 1.0
        self.parent = parent
        self.action = action

    def puct(self, c=1.0):
        if self.visits == 0:
            return float('inf')
        q = self.total_value / self.visits
        u = c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u


# ─────────────────────────────────────────────────────────────────────────────
#  Agent chính
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

        # Load model
        self.model = BomberNet(INPUT_CHANNELS)
        model_path = os.path.join(os.path.dirname(__file__), "weights.pth")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def act(self, obs):
        if obs["players"][self.agent_id][2] == 0:
            return STOP
        alive = sum(1 for p in obs["players"] if p[2] == 1)
        action = self._mcts_act(obs) if alive <= 2 else self._policy_act(obs)
        self.step_count += 1
        return action

    def _policy_act(self, obs):
        """Dùng policy trực tiếp, che hành động không hợp lệ."""
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        my_id   = self.agent_id

        state_tensor = encode_obs(grid, players, bombs, my_id, self.step_count).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)

        legal = self._legal_actions(grid, players, bombs, my_id)
        if not legal:
            return STOP

        mask = torch.full((6,), -float('inf'))
        mask[list(legal)] = 0.0
        masked_logits = logits + mask
        action = int(torch.argmax(masked_logits).item())
        return action

    def _mcts_act(self, obs, budget=0.085):
        """MCTS với prior từ policy. Dùng cho 1v1."""
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]
        my_id   = self.agent_id
        alive_ids = [i for i, p in enumerate(players) if p[2] == 1]
        if len(alive_ids) < 2:
            return self._policy_act(obs)
        enemy_id = next(i for i in alive_ids if i != my_id)

        # Khởi tạo trạng thái mô phỏng
        init_state = SimState(
            grid.astype(np.int8),
            np.array([[int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])] for p in players]),
            bombs.astype(np.int8) if len(bombs)>0 else np.empty((0,4), dtype=np.int8)
        )

        # Tính prior từ mạng cho root
        state_tensor = encode_obs(grid, players, bombs, my_id, self.step_count).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0).numpy()
        legal = self._legal_actions(grid, players, bombs, my_id)
        # chuyển logits sang xác suất (softmax), chỉ trên các hành động hợp lệ
        probs = np.zeros(6)
        exp = np.exp(logits[list(legal)] - np.max(logits[list(legal)]))
        sum_exp = exp.sum()
        for idx, a in enumerate(legal):
            probs[a] = exp[idx] / sum_exp if sum_exp > 0 else 1.0/len(legal)

        root = _MCTSNode(untried=legal, prior=1.0)  # prior sẽ được set sau
        root.prior = 1.0  # không quan trọng cho root
        # Tạo trước các nút con với prior tương ứng
        for a in legal:
            root.children[a] = _MCTSNode(untried=[],
                                         prior=probs[a],
                                         parent=root,
                                         action=a)
            root.children[a].untried = self._legal_actions_sim(init_state, my_id)  # sẽ thay đổi khi đi xuống
        root.visits = 1

        deadline = time.time() + budget
        while time.time() < deadline:
            node = root
            state = init_state.copy()
            path = [node]

            # Selection
            while not state.is_terminal() and node.untried == [] and node.children:
                # chọn con tốt nhất theo puct
                best_a = max(node.children, key=lambda a: node.children[a].puct())
                node = node.children[best_a]
                path.append(node)
                # mô phỏng bước đi
                enemy_act = self._heuristic_enemy_act(state, enemy_id, my_id)
                state.step({my_id: best_a, enemy_id: enemy_act})

            # Expansion & Evaluation
            if not state.is_terminal():
                if node.untried:
                    action = node.untried.pop()
                    enemy_act = self._heuristic_enemy_act(state, enemy_id, my_id)
                    state.step({my_id: action, enemy_id: enemy_act})
                    # tính prior cho nút mới từ mạng
                    legal_new = self._legal_actions_sim(state, my_id)
                    if legal_new:
                        # encode state để lấy policy
                        # Chúng ta cần encode từ state.grid, state.players, state.bombs
                        obs_new = {
                            "map": state.grid,
                            "players": state.players,
                            "bombs": state.bombs if len(state.bombs)>0 else np.empty((0,4), dtype=np.int8)
                        }
                        # Ở đây step tạm để 0 vì mô phỏng
                        s_tensor = encode_obs(state.grid, state.players, state.bombs, my_id, 0).unsqueeze(0)
                        with torch.no_grad():
                            logits_new = self.model(s_tensor).squeeze(0).numpy()
                        exp_new = np.exp(logits_new[list(legal_new)] - np.max(logits_new[list(legal_new)]))
                        sum_new = exp_new.sum()
                        for idx, a in enumerate(legal_new):
                            prior = (exp_new[idx] / sum_new) if sum_new > 0 else 1.0/len(legal_new)
                            child = _MCTSNode(untried=[], prior=prior, parent=node, action=a)
                            node.children[a] = child
                    else:
                        child = _MCTSNode(untried=[], prior=1.0, parent=node, action=action)
                        node.children[action] = child
                    node = child
                    path.append(node)
                else:
                    # leaf node, evaluate with rollout
                    pass

            # Rollout & Backprop
            if not state.is_terminal():
                value = self._rollout(state, my_id, enemy_id, depth=15)
            else:
                value = self._terminal_value(state, my_id, enemy_id)

            for n in path[::-1]:
                n.visits += 1
                n.total_value += value
                value = -value   # đổi phe (do zero-sum)

        if not root.children:
            return self._policy_act(obs)
        best_a = max(root.children, key=lambda a: root.children[a].visits)
        return best_a

    def _rollout(self, state, my_id, enemy_id, depth):
        s = state.copy()
        disc = 1.0
        total = 0.0
        for _ in range(depth):
            if s.is_terminal() or s.players[my_id][2]!=1 or s.players[enemy_id][2]!=1:
                break
            my_act = self._heuristic_rollout_act(s, my_id, enemy_id)
            en_act = self._heuristic_rollout_act(s, enemy_id, my_id)
            rewards = s.step({my_id: my_act, enemy_id: en_act})
            total += disc * rewards.get(my_id, 0.0)
            disc *= 0.95
        # terminal bonus
        my_alive = s.players[my_id][2]==1
        en_alive = s.players[enemy_id][2]==1
        if my_alive and not en_alive:
            total += disc * 600.0
        elif not my_alive and en_alive:
            total -= disc * 400.0
        return total

    def _terminal_value(self, state, my_id, enemy_id):
        my_alive = state.players[my_id][2]==1
        en_alive = state.players[enemy_id][2]==1
        if my_alive and not en_alive: return 1.0
        if not my_alive and en_alive: return -1.0
        return 0.0

    def _heuristic_enemy_act(self, state, eid, mid):
        """Đối thủ đơn giản: né bom, đuổi theo ta, thả bom nếu lợi."""
        p = state.players[eid]
        if p[2] != 1:
            return STOP
        pos = (int(p[0]), int(p[1]))
        bomb_pos = {(b[0], b[1]) for b in state.bombs}
        dnow, dsoon = state.danger_tiles()

        # escape
        if pos in dnow or pos in dsoon:
            for a in random.sample([LEFT, RIGHT, UP, DOWN], 4):
                nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a
            return STOP

        # bomb if can hit us
        my_pos = (int(state.players[mid][0]), int(state.players[mid][1]))
        r = max(1, p[4]+1)
        if p[3] > 0 and pos not in bomb_pos:
            blast = state.blast_tiles(pos[0], pos[1], r)
            if my_pos in blast and self._has_escape(state, pos, dsoon | blast):
                return BOMB

        # move toward us
        dx = my_pos[0] - pos[0]
        dy = my_pos[1] - pos[1]
        cands = [RIGHT if dx>0 else LEFT, DOWN if dy>0 else UP] if abs(dx)>=abs(dy) else [DOWN if dy>0 else UP, RIGHT if dx>0 else LEFT]
        for a in cands:
            nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
            if state.passable(nx, ny) and (nx, ny) not in dsoon:
                return a
        return STOP

    def _heuristic_rollout_act(self, state, pid, oid):
        """Rollout policy nhanh."""
        p = state.players[pid]
        if p[2] != 1:
            return STOP
        pos = (int(p[0]), int(p[1]))
        bp = {(b[0], b[1]) for b in state.bombs}
        dnow, dsoon = state.danger_tiles()
        if pos in dnow or pos in dsoon:
            for a in random.sample([LEFT, RIGHT, UP, DOWN], 4):
                nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a
            return STOP
        # grab items
        for a in [LEFT, RIGHT, UP, DOWN]:
            nx, ny = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
            if state.passable(nx, ny) and (nx, ny) not in dsoon and int(state.grid[nx, ny]) in (3,4):
                return a
        # bomb opponent if nearby
        opp = state.players[oid]
        if opp[2]==1:
            r = max(1, p[4]+1)
            if p[3]>0 and pos not in bp:
                blast = state.blast_tiles(pos[0], pos[1], r)
                if (int(opp[0]), int(opp[1])) in blast and self._has_escape(state, pos, dsoon|blast):
                    return BOMB
        safe = [a for a in [LEFT,RIGHT,UP,DOWN,STOP] if a==STOP or (state.passable(*(pos[0]+MOVES[a][0], pos[1]+MOVES[a][1])) and (pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]) not in dsoon)]
        return random.choice(safe) if safe else STOP

    def _has_escape(self, state, pos, danger):
        q, seen = deque([(pos,0)]), {pos}
        while q:
            p, d = q.popleft()
            if p not in danger and d > 0:
                return True
            if d >= 7:
                continue
            for a in [LEFT, RIGHT, UP, DOWN]:
                nx, ny = p[0]+MOVES[a][0], p[1]+MOVES[a][1]
                npos = (nx, ny)
                if npos not in seen and state.passable(nx, ny):
                    seen.add(npos)
                    q.append((npos, d+1))
        return False

    def _legal_actions(self, grid, players, bombs, my_id):
        me = players[my_id]
        if me[2] == 0:
            return [STOP]
        x, y = int(me[0]), int(me[1])
        bp = {(b[0], b[1]) for b in bombs}
        acts = [STOP]
        for a in (LEFT, RIGHT, UP, DOWN):
            nx, ny = x+MOVES[a][0], y+MOVES[a][1]
            if _passable(grid, nx, ny) and (nx, ny) not in bp:
                acts.append(a)
        if me[3] > 0 and (x, y) not in bp:
            acts.append(BOMB)
        return acts

    def _legal_actions_sim(self, state, pid):
        p = state.players[pid]
        if p[2] == 0:
            return [STOP]
        x, y = int(p[0]), int(p[1])
        bp = {(b[0], b[1]) for b in state.bombs}
        acts = [STOP]
        for a in (LEFT, RIGHT, UP, DOWN):
            nx, ny = x+MOVES[a][0], y+MOVES[a][1]
            if state.passable(nx, ny) and (nx, ny) not in bp:
                acts.append(a)
        if p[3] > 0 and (x, y) not in bp:
            acts.append(BOMB)
        return acts