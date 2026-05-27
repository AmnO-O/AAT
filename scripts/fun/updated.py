import os
import sys
import json
import random
from collections import Counter, deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# -----------------------------------------------------------------------------
# Local contest imports
# -----------------------------------------------------------------------------
sys.path.append(os.getcwd())
from engine.game import BomberEnv
from agent.tactical_rule_agent import TacticalRuleAgent
from agent.genius_rule_agent import GeniusRuleAgent
from agent.smarter_rule_agent import SmarterRuleAgent
from agent.random_agent import RandomAgent

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BOARD_SIZE = 13
INPUT_CHANNELS = 20
NUM_ACTIONS = 6
MAX_STEPS = 500

INITIAL_GAMES = 100
DAGGER_ROUNDS = 2
DAGGER_GAMES_PER_ROUND = 200

TRAIN_SPLIT_MOD = 10   # seed % 10 == 0 -> validation
CHUNK_SIZE = 2048
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRAD_CLIP_NORM = 1.0

TRAIN_DIR = "bc_train_chunks"
VAL_DIR = "bc_val_chunks"
MODEL_PATH = "model_bc.pth"
BEST_MODEL_PATH = "model_bc_best.pth"
MANIFEST_NAME = "manifest.json"

TEACHER_MODE = "tactical"  # "tactical" or "ensemble"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

# -----------------------------------------------------------------------------
# Board utilities
# -----------------------------------------------------------------------------

MOVES = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


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


def bfs_reachable_count(
    grid: np.ndarray,
    start: Tuple[int, int],
    bombs: np.ndarray,
    max_depth: int = 3,
) -> int:
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
# Observation encoding
# -----------------------------------------------------------------------------

def encode_obs(grid: np.ndarray, players: np.ndarray, bombs: np.ndarray, my_id: int, step: int) -> torch.Tensor:
    """Return (C, H, W) tensor. my_id is dynamic; do not hard-code a seat."""
    C, H, W = INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE
    state = np.zeros((C, H, W), dtype=np.float32)

    # Static map planes
    state[0] = (grid == 1).astype(np.float32)
    state[1] = (grid == 2).astype(np.float32)
    state[2] = (grid == 0).astype(np.float32)
    state[3] = (grid == 3).astype(np.float32)
    state[4] = (grid == 4).astype(np.float32)

    # Player planes by actual id
    for pid in range(4):
        if pid < len(players) and int(players[pid][2]) == 1:
            r, c = int(players[pid][0]), int(players[pid][1])
            if in_bounds(r, c):
                state[5 + pid, r, c] = 1.0

    # Danger plane: bombs that will explode immediately / next step
    state[9] = danger_plane(grid, players, bombs, timer_threshold=1)

    # Ego features
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

    # Bomb timers
    for b in bombs:
        r, c, timer = int(b[0]), int(b[1]), int(b[2])
        if in_bounds(r, c):
            state[12, r, c] = 1.0 / float(timer + 1)

    state[13] = float(bomb_radius) / 6.0

    # BFS distances (stronger than Manhattan because it respects walls/boxes/bombs)
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
    state[19] = float(me_alive)   # a simple alive flag; not global alive count

    return torch.from_numpy(state)


# -----------------------------------------------------------------------------
# Teacher policy
# -----------------------------------------------------------------------------

class TeacherPolicy:
    def __init__(self, agent_id: int):
        self.tactical = TacticalRuleAgent(agent_id)
        self.genius = GeniusRuleAgent(agent_id)
        self.smarter = SmarterRuleAgent(agent_id)

    def act(self, obs: Dict) -> int:
        if TEACHER_MODE == "tactical":
            return int(self.tactical.act(obs))

        # Ensemble fallback: tactical priority, then majority vote
        acts = [
            int(self.tactical.act(obs)),
            int(self.genius.act(obs)),
            int(self.smarter.act(obs)),
        ]
        vote = Counter(acts)
        best_count = max(vote.values())
        candidates = [a for a, c in vote.items() if c == best_count]
        if acts[0] in candidates:
            return acts[0]
        return int(candidates[0])


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
        out = torch.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
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
# Augmentation
# -----------------------------------------------------------------------------

def remap_action_horizontal(action: int) -> int:
    return {1: 2, 2: 1, 3: 3, 4: 4, 0: 0, 5: 5}.get(int(action), int(action))


def remap_action_vertical(action: int) -> int:
    return {3: 4, 4: 3, 1: 1, 2: 2, 0: 0, 5: 5}.get(int(action), int(action))


def augment_tensor_and_action(state: torch.Tensor, action: int) -> Tuple[torch.Tensor, int]:
    p = random.random()
    if p < 0.33:
        state = torch.flip(state, dims=[2])  # horizontal
        action = remap_action_horizontal(action)
    elif p < 0.66:
        state = torch.flip(state, dims=[1])  # vertical
        action = remap_action_vertical(action)
    else:
        state = torch.flip(state, dims=[1, 2])  # 180 degree
        action = remap_action_vertical(remap_action_horizontal(action))
    return state, int(action)


# -----------------------------------------------------------------------------
# Chunk helpers
# -----------------------------------------------------------------------------

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



def flush_chunk(chunk_dir: str, chunk_idx: int, states: List[np.ndarray], actions: List[int], seeds: List[int]) -> Dict:
    if not states:
        return {}
    states_np = np.stack(states, axis=0).astype(np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    seeds_np = np.array(seeds, dtype=np.int64)
    hist = np.bincount(actions_np, minlength=NUM_ACTIONS).astype(int).tolist()

    filename = f"chunk_{chunk_idx:05d}.npz"
    file_path = os.path.join(chunk_dir, filename)
    np.savez_compressed(file_path, states=states_np, actions=actions_np, seeds=seeds_np)

    return {
        "file": filename,
        "count": int(len(actions_np)),
        "action_hist": hist,
        "seed_min": int(seeds_np.min()) if len(seeds_np) else None,
        "seed_max": int(seeds_np.max()) if len(seeds_np) else None,
    }


# -----------------------------------------------------------------------------
# Dataset: lazy, chunk-wise streaming
# -----------------------------------------------------------------------------

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
        rng = np.random.default_rng(self.seed)
        chunk_indices = np.arange(len(self.chunks))
        if self.shuffle_chunks:
            rng.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            chunk_meta = self.chunks[int(chunk_idx)]
            file_path = os.path.join(self.chunk_dir, chunk_meta["file"])
            data = np.load(file_path)
            states = data["states"]
            actions = data["actions"]

            idxs = np.arange(len(actions))
            if self.shuffle_within_chunk:
                rng.shuffle(idxs)

            for i in idxs:
                state = torch.from_numpy(states[int(i)]).float()
                action = int(actions[int(i)])
                if self.augment:
                    state, action = augment_tensor_and_action(state, action)
                yield state, torch.tensor(action, dtype=torch.long)


# -----------------------------------------------------------------------------
# Balanced class weights
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Opponent setup
# -----------------------------------------------------------------------------

def build_opponents(controlled_id: int, game_seed: int) -> Dict[int, object]:
    """Return one baseline agent for each non-controlled slot."""
    rng = random.Random(game_seed)
    pool = [TacticalRuleAgent, SmarterRuleAgent, GeniusRuleAgent, RandomAgent]
    chosen = rng.sample(pool, 3)
    other_ids = [pid for pid in range(4) if pid != controlled_id]
    opponents = {}
    for pid, cls in zip(other_ids, chosen):
        opponents[pid] = cls(pid)
    return opponents


# -----------------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------------

def collect_one_step(obs: Dict, teacher: TeacherPolicy, my_id: int, step: int) -> Tuple[np.ndarray, int]:
    state = encode_obs(obs["map"], obs["players"], obs["bombs"], my_id, step).numpy().astype(np.float32)
    action = int(teacher.act(obs))
    return state, action



def collect_initial_data(train_dir: str, val_dir: str, num_games: int) -> None:
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    train_manifest = load_manifest(train_dir)
    val_manifest = load_manifest(val_dir)
    train_chunk_idx = len(train_manifest.get("chunks", []))
    val_chunk_idx = len(val_manifest.get("chunks", []))

    train_buf_states, train_buf_actions, train_buf_seeds = [], [], []
    val_buf_states, val_buf_actions, val_buf_seeds = [], [], []

    for game_idx in range(num_games):
        seed = SEED + game_idx
        controlled_id = game_idx % 4
        split = "val" if (seed % TRAIN_SPLIT_MOD == 0) else "train"

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherPolicy(controlled_id)
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            state_np, expert_action = collect_one_step(obs, teacher, controlled_id, step)

            if split == "train":
                train_buf_states.append(state_np)
                train_buf_actions.append(expert_action)
                train_buf_seeds.append(seed)
            else:
                val_buf_states.append(state_np)
                val_buf_actions.append(expert_action)
                val_buf_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = expert_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            if split == "train" and len(train_buf_states) >= CHUNK_SIZE:
                entry = flush_chunk(train_dir, train_chunk_idx, train_buf_states, train_buf_actions, train_buf_seeds)
                if entry:
                    train_manifest["chunks"].append(entry)
                    save_manifest(train_dir, train_manifest)
                    train_chunk_idx += 1
                train_buf_states.clear()
                train_buf_actions.clear()
                train_buf_seeds.clear()

            if split == "val" and len(val_buf_states) >= CHUNK_SIZE:
                entry = flush_chunk(val_dir, val_chunk_idx, val_buf_states, val_buf_actions, val_buf_seeds)
                if entry:
                    val_manifest["chunks"].append(entry)
                    save_manifest(val_dir, val_manifest)
                    val_chunk_idx += 1
                val_buf_states.clear()
                val_buf_actions.clear()
                val_buf_seeds.clear()

        if (game_idx + 1) % 100 == 0:
            train_count = sum(c["count"] for c in train_manifest.get("chunks", [])) + len(train_buf_actions)
            val_count = sum(c["count"] for c in val_manifest.get("chunks", [])) + len(val_buf_actions)
            print(f"Collected {game_idx + 1}/{num_games} games | train={train_count} | val={val_count}", flush=True)

    if train_buf_states:
        entry = flush_chunk(train_dir, train_chunk_idx, train_buf_states, train_buf_actions, train_buf_seeds)
        if entry:
            train_manifest["chunks"].append(entry)
    if val_buf_states:
        entry = flush_chunk(val_dir, val_chunk_idx, val_buf_states, val_buf_actions, val_buf_seeds)
        if entry:
            val_manifest["chunks"].append(entry)

    save_manifest(train_dir, train_manifest)
    save_manifest(val_dir, val_manifest)


# -----------------------------------------------------------------------------
# DAgger: collect student mistakes and relabel with teacher
# -----------------------------------------------------------------------------

def collect_dagger_data(model: nn.Module, out_dir: str, num_games: int) -> int:
    ensure_dir(out_dir)
    model.eval()

    out_manifest = load_manifest(out_dir)
    chunk_idx = len(out_manifest.get("chunks", []))
    buf_states, buf_actions, buf_seeds = [], [], []
    collected = 0

    def flush_buffer():
        nonlocal chunk_idx, collected
        if not buf_states:
            return
        entry = flush_chunk(out_dir, chunk_idx, buf_states, buf_actions, buf_seeds)
        if entry:
            out_manifest["chunks"].append(entry)
            save_manifest(out_dir, out_manifest)
            collected += entry["count"]
            chunk_idx += 1
        buf_states.clear()
        buf_actions.clear()
        buf_seeds.clear()

    for game_idx in range(num_games):
        seed = 100000 + SEED + game_idx
        controlled_id = game_idx % 4

        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        teacher = TeacherPolicy(controlled_id)
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(state)
                student_action = int(torch.argmax(logits, dim=1).item())

            expert_action = int(teacher.act(obs))
            if student_action != expert_action or (student_action == 0 and expert_action != 0):
                buf_states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
                buf_actions.append(expert_action)
                buf_seeds.append(seed)

            actions = [0, 0, 0, 0]
            actions[controlled_id] = student_action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            if len(buf_states) >= CHUNK_SIZE:
                flush_buffer()

        if (game_idx + 1) % 50 == 0:
            print(f"DAgger {game_idx + 1}/{num_games} games | new samples={collected + len(buf_actions)}", flush=True)

    flush_buffer()
    return collected


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def build_loaders(train_dir: str, val_dir: str):
    train_ds = ChunkedBomberIterableDataset(
        train_dir,
        augment=True,
        shuffle_chunks=True,
        shuffle_within_chunk=True,
        seed=SEED,
    )
    val_ds = ChunkedBomberIterableDataset(
        val_dir,
        augment=False,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        seed=SEED,
    )

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

        total_loss += float(loss.item()) * states.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == actions).sum().item())
        total_count += int(states.size(0))

        if batch_idx % 50 == 0:
            print(f"  batch={batch_idx} / ~{len(loader)}", flush=True)
        
        print(f"total_loss={total_loss:.4f} total_acc={total_correct / max(1, total_count):.4f}", flush=True)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc



def train_policy_model(train_dir: str, val_dir: str, init_model_path: str | None = None, lr: float = LEARNING_RATE):
    train_loader, val_loader, class_weights = build_loaders(train_dir, val_dir)

    model = BomberNet(INPUT_CHANNELS).to(DEVICE)
    if init_model_path and os.path.exists(init_model_path):
        model.load_state_dict(torch.load(init_model_path, map_location=DEVICE))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d}/{EPOCHS}", flush=True)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
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


# -----------------------------------------------------------------------------
# Optional sanity eval
# -----------------------------------------------------------------------------

def quick_eval_against_baselines(model: nn.Module, num_games: int = 20) -> None:
    model.eval()
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(num_games):
        seed = 200000 + SEED + game_idx
        controlled_id = game_idx % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=seed)
        obs = env.reset()
        opponents = build_opponents(controlled_id, seed)

        done = False
        step = 0
        while not done:
            with torch.no_grad():
                state = encode_obs(obs["map"], obs["players"], obs["bombs"], controlled_id, step).unsqueeze(0).to(DEVICE)
                logits = model(state)
                action = int(torch.argmax(logits, dim=1).item())

            actions = [0, 0, 0, 0]
            actions[controlled_id] = action
            for pid, agent in opponents.items():
                actions[pid] = int(agent.act(obs))

            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

        alive = [int(p[2]) for p in obs["players"]]
        my_alive = alive[controlled_id]
        alive_count = sum(alive)
        if my_alive == 1 and alive_count == 1:
            wins += 1
        elif my_alive == 1:
            draws += 1
        else:
            losses += 1

    print(f"Quick eval proxy | wins={wins} draws={draws} losses={losses}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    print("=== Phase 1: Collect initial demonstrations ===", flush=True)
    collect_initial_data(TRAIN_DIR, VAL_DIR, INITIAL_GAMES)

    print("=== Phase 2: Train initial policy ===", flush=True)
    model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=None, lr=LEARNING_RATE)

    for round_idx in range(DAGGER_ROUNDS):
        print(f"=== Phase 3.{round_idx + 1}: DAgger collection ===", flush=True)
        new_samples = collect_dagger_data(model, TRAIN_DIR, DAGGER_GAMES_PER_ROUND)
        print(f"DAgger round {round_idx + 1}: collected {new_samples} correction samples", flush=True)

        print(f"=== Phase 4.{round_idx + 1}: Fine-tune with aggregated data ===", flush=True)
        model = train_policy_model(TRAIN_DIR, VAL_DIR, init_model_path=MODEL_PATH, lr=FINE_TUNE_LR)

    print("=== Optional quick sanity check ===", flush=True)
    quick_eval_against_baselines(model, num_games=20)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
