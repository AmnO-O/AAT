import numpy as np
import torch
import torch.nn as nn
import random
import math
import time

# ---------- Constants ----------
STOP    = 0
LEFT    = 1
RIGHT   = 2
UP      = 3
DOWN    = 4
BOMB    = 5

MOVES = {
    STOP:  (0, 0),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
    UP:    (0, -1),
    DOWN:  (0, 1),
}

GRASS = 0
WALL  = 1
BOX   = 2
ITEM_RADIUS   = 3
ITEM_CAPACITY = 4

# ---------- Mạng nơ-ron (giống lúc train) ----------
class BomberNet(nn.Module):
    def __init__(self, input_channels=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, 6)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value

# ---------- Mã hóa trạng thái ----------
def encode_obs(grid, players, bombs, my_id):
    C, H, W = 10, 13, 13
    state = np.zeros((C, H, W), dtype=np.float32)
    state[0] = (grid == 1)
    state[1] = (grid == 2)
    state[2] = (grid == 0)
    state[3] = (grid == 3)
    state[4] = (grid == 4)
    for pid in range(4):
        if pid < len(players) and players[pid][2] == 1:
            x, y = int(players[pid][0]), int(players[pid][1])
            if 0 <= x < H and 0 <= y < W:
                state[5 + pid, x, y] = 1.0
    danger = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        owner = int(b[3]) if len(b) > 3 else -1
        radius = 1
        if 0 <= owner < len(players):
            radius = 1 + int(players[owner][4])
        if timer <= 1:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                for r in range(1, radius+1):
                    x, y = bx + dx*r, by + dy*r
                    if x < 0 or x >= H or y < 0 or y >= W:
                        break
                    if grid[x, y] == 1:
                        break
                    danger[x, y] = 1.0
                    if grid[x, y] == 2:
                        break
    state[9] = danger
    return torch.from_numpy(state).unsqueeze(0)

# ---------- Lớp trạng thái game cho MCTS ----------
class GameState:
    def __init__(self, grid, players, bombs, step, my_id, rng):
        self.grid = grid.copy()
        self.players = players.copy()
        self.bombs = bombs.copy() if len(bombs) > 0 else np.empty((0,4), dtype=np.int8)
        self.step = step
        self.my_id = my_id
        self.rng = rng

    def get_legal_actions(self):
        me = self.players[self.my_id]
        if me[2] == 0:
            return [STOP]
        x, y = me[0], me[1]
        bomb_set = {(b[0], b[1]) for b in self.bombs}
        actions = [STOP]
        for a in (LEFT, RIGHT, UP, DOWN):
            nx, ny = x + MOVES[a][0], y + MOVES[a][1]
            if 0 <= nx < 13 and 0 <= ny < 13 and self.grid[nx, ny] in (GRASS, ITEM_RADIUS, ITEM_CAPACITY):
                if (nx, ny) not in bomb_set:
                    actions.append(a)
        if me[3] > 0 and (x, y) not in bomb_set:
            actions.append(BOMB)
        return actions

    def apply_action(self, action):
        # Sao chép mọi thứ
        grid = self.grid.copy()
        players = self.players.copy()
        bombs = self.bombs.copy() if len(self.bombs) > 0 else np.empty((0,4), dtype=np.int8)
        step = self.step + 1
        rng = self.rng

        # Đối thủ: chọn ngẫu nhiên trong các hành động hợp lệ
        enemy_actions = {}
        for pid in range(4):
            if pid != self.my_id and players[pid][2] == 1:
                ex, ey = players[pid][0], players[pid][1]
                e_acts = [STOP]
                for a in (LEFT, RIGHT, UP, DOWN):
                    nx, ny = ex + MOVES[a][0], ey + MOVES[a][1]
                    if 0 <= nx < 13 and 0 <= ny < 13 and grid[nx, ny] in (GRASS, ITEM_RADIUS, ITEM_CAPACITY):
                        if (nx, ny) not in {(b[0], b[1]) for b in bombs}:
                            e_acts.append(a)
                if players[pid][3] > 0 and (ex, ey) not in {(b[0], b[1]) for b in bombs}:
                    e_acts.append(BOMB)
                enemy_actions[pid] = rng.choice(e_acts)

        # Tổng hợp hành động
        all_actions = {}
        for pid in range(4):
            if pid == self.my_id:
                all_actions[pid] = action
            else:
                all_actions[pid] = enemy_actions.get(pid, STOP)

        # Thực hiện di chuyển
        new_pos = {}
        for pid in range(4):
            if players[pid][2] == 1:
                old_x, old_y = players[pid][0], players[pid][1]
                act = all_actions[pid]
                if act in (LEFT, RIGHT, UP, DOWN):
                    dx, dy = MOVES[act]
                    nx, ny = old_x+dx, old_y+dy
                    if 0 <= nx < 13 and 0 <= ny < 13 and grid[nx, ny] in (GRASS, ITEM_RADIUS, ITEM_CAPACITY):
                        if (nx, ny) not in {(b[0], b[1]) for b in bombs}:
                            players[pid][0], players[pid][1] = nx, ny
                new_pos[pid] = (players[pid][0], players[pid][1])

        # Đặt bom
        for pid in range(4):
            if all_actions[pid] == BOMB and players[pid][2] == 1 and players[pid][3] > 0:
                x, y = players[pid][0], players[pid][1]
                if (x, y) not in {(b[0], b[1]) for b in bombs}:
                    new_bomb = np.array([[x, y, 7, pid]], dtype=np.int8)
                    if len(bombs) == 0:
                        bombs = new_bomb
                    else:
                        bombs = np.vstack([bombs, new_bomb])
                    players[pid][3] -= 1

        # Giảm timer bom
        for b in bombs:
            b[2] -= 1

        # Xử lý nổ (có chain reaction)
        exploded_mask = np.zeros(len(bombs), dtype=bool)
        while True:
            new_explosions = False
            blast_tiles = set()
            for i, b in enumerate(bombs):
                if exploded_mask[i]:
                    continue
                if b[2] <= 0:
                    bx, by, _, owner = b
                    radius = 1 + int(players[owner][4])
                    # Tính ô bị nổ
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        for r in range(1, radius+1):
                            x, y = bx + dx*r, by + dy*r
                            if x < 0 or x >= 13 or y < 0 or y >= 13:
                                break
                            cell = grid[x, y]
                            if cell == WALL:
                                break
                            blast_tiles.add((x, y))
                            if cell == BOX:
                                break
                    blast_tiles.add((bx, by))
                    exploded_mask[i] = True
                    new_explosions = True
            if not new_explosions:
                break

            # Giết agent
            for pid in range(4):
                if players[pid][2] == 1 and (players[pid][0], players[pid][1]) in blast_tiles:
                    players[pid][2] = 0

            # Phá hộp, rơi item
            for (x, y) in blast_tiles:
                if grid[x, y] == BOX:
                    grid[x, y] = GRASS
                    r = rng.random()
                    if r < 0.3:
                        grid[x, y] = ITEM_RADIUS
                    elif r < 0.6:
                        grid[x, y] = ITEM_CAPACITY

            # Dây chuyền: bom nào nằm trong vùng nổ sẽ nổ ngay
            for b in bombs:
                if not exploded_mask[(b == bombs).all(axis=1)]:
                    if (b[0], b[1]) in blast_tiles:
                        b[2] = 0

        # Loại bỏ bom đã nổ
        bombs = bombs[~exploded_mask]
        if len(bombs) == 0:
            bombs = np.empty((0,4), dtype=np.int8)

        # Nhặt item
        for pid in range(4):
            if players[pid][2] == 1:
                x, y = players[pid][0], players[pid][1]
                if grid[x, y] == ITEM_RADIUS:
                    players[pid][4] = min(4, players[pid][4] + 1)
                    grid[x, y] = GRASS
                elif grid[x, y] == ITEM_CAPACITY:
                    players[pid][3] = min(5, players[pid][3] + 1)
                    grid[x, y] = GRASS

        return GameState(grid, players, bombs, step, self.my_id, rng)

    def is_terminal(self):
        alive = sum(1 for p in self.players if p[2] == 1)
        return self.step >= 500 or self.players[self.my_id][2] == 0 or alive <= 1

    def get_reward(self):
        if self.players[self.my_id][2] == 0:
            return -1.0
        if sum(1 for p in self.players if p[2] == 1) <= 1:
            return 1.0
        return 0.0

# ---------- Nút MCTS ----------
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_value = 0.0

    def is_expanded(self):
        return len(self.children) > 0

# ---------- Agent chính ----------
class AgentBase:
    team_id = "Samnu"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.model = BomberNet(input_channels=10)
        # Load model đã train
        self.model.load_state_dict(torch.load("model_bc.pth", map_location="cpu"))
        self.model.eval()
        self.rng = random.Random()

    def act(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        my_id = self.agent_id

        if players[my_id][2] == 0:
            return STOP

        root_state = GameState(grid, players, bombs, 0, my_id, self.rng)
        root = MCTSNode(root_state)

        start_time = time.time()
        sims = 0
        # Chạy tối đa 50 mô phỏng hoặc đến khi hết 90ms
        while sims < 50 and time.time() - start_time < 0.09:
            node = root
            state = root_state

            # Selection
            while node.is_expanded() and not state.is_terminal():
                best_child = None
                best_ucb = -float('inf')
                for child in node.children.values():
                    if child.visits == 0:
                        ucb = float('inf')
                    else:
                        ucb = child.total_value / child.visits + 1.4 * math.sqrt(math.log(node.visits) / child.visits)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                node = best_child
                state = node.state

            # Expansion & Evaluation
            if not state.is_terminal():
                legal = state.get_legal_actions()
                # Dùng mạng để đánh giá trạng thái hiện tại
                state_tensor = encode_obs(state.grid, state.players, state.bombs, self.agent_id)
                with torch.no_grad():
                    logits, value = self.model(state_tensor)
                value = value.item()
                # Thêm một nút con cho một hành động chưa thử
                for a in legal:
                    if a not in node.children:
                        next_state = state.apply_action(a)
                        child = MCTSNode(next_state, parent=node, action=a)
                        node.children[a] = child
                        break
                # Backpropagation
                while node is not None:
                    node.visits += 1
                    node.total_value += value
                    node = node.parent
            else:
                reward = state.get_reward()
                while node is not None:
                    node.visits += 1
                    node.total_value += reward
                    node = node.parent
            sims += 1

        # Chọn hành động được thăm nhiều nhất
        best_action = None
        max_visits = -1
        for act, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = act
        return best_action if best_action is not None else STOP