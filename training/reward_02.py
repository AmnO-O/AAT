"""ICEC 2019 Bomberman-style rewards (Goulart et al., Table 1).

Scaled to this engine's observations. Kill credit is approximate: the env does
not attribute deaths to a specific bomb owner, so we use +1.0 per net enemy
death while the learner remains alive (same pragmatic choice as reward.py).

Reference: Learning How to Play Bomberman with Deep RL and IL (hal-03652029).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine import Map

from training.reward import (
    _any_bombs,
    _blast_status_at,
    _bomb_radius_from_obs,
    _enemy_alive_count,
    _explosion_tiles_for_bomb,
    _manhattan_to_nearest_alive_enemy,
    _parse_bomb_row,
)

# Table 1 (paper) — numeric values as published
R_AGENT_DEATH = -0.5
R_ENEMY_DEATH = 1.0
R_LAST_ALIVE = 1.0
R_BLOCK_DESTROYED = 0.1
R_CLOSEST_SO_FAR = 0.1
R_CLOSER_STEP = 0.002
R_FARTHER_STEP = -0.002
R_TIME_STEP = -0.01
R_IN_BLAST = -0.000666
R_SAFE_NEAR_BOMB = 0.002

# Manhattan distance to nearest bomb center to count as "bomb nearby" when safe
SAFE_NEAR_BOMB_RADIUS = 4

# Anti–reward-hacking: dense shaping cap per episode, decay with time, and
# movement bonuses only when improving global best distance to an enemy.
DENSE_SHAPING_CAP_PER_EPISODE = 0.45
DENSE_DECAY_TAU_STEPS = 280.0


@dataclass
class EpisodeRewardState:
    """Mutable per-episode state for closest-enemy-so-far shaping."""

    best_dist: Optional[float] = None
    step_idx: int = 0
    dense_cumulative: float = 0.0

    def reset(self) -> None:
        self.best_dist = None
        self.step_idx = 0
        self.dense_cumulative = 0.0


def _box_count(obs) -> int:
    return int(np.sum(obs["map"] == Map.BOX))


def _min_manhattan_to_bomb_center(obs, x: int, y: int) -> Optional[int]:
    bombs = obs["bombs"]
    if bombs is None or np.asarray(bombs).size == 0:
        return None
    arr = np.asarray(bombs)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    best = None
    for i in range(arr.shape[0]):
        parsed = _parse_bomb_row(arr[i])
        if parsed is None:
            continue
        bx, by, _, _ = parsed
        d = abs(int(x) - int(bx)) + abs(int(y) - int(by))
        best = d if best is None else min(best, d)
    return best


def compute_reward_icec(
    prev_obs,
    curr_obs,
    agent_id: int,
    episode_state: EpisodeRewardState,
    out_components: Optional[Dict[str, float]] = None,
) -> Tuple[float, EpisodeRewardState]:
    """Return (reward, episode_state). Mutates ``episode_state`` in place.

    Movement shaping (closer / closest-so-far) only applies when the step
    **improves** the running best Manhattan distance to a living enemy, so
    oscillating left/right cannot farm dense rewards. Positive dense shaping is
    decayed over time and capped per episode.

    If ``out_components`` is a dict, it is cleared and filled with this step's
    contribution per component (before episode cap; ``dense_applied`` is after).
    """
    if prev_obs is None:
        if out_components is not None:
            out_components.clear()
        return 0.0, episode_state

    episode_state.step_idx += 1
    decay = math.exp(-episode_state.step_idx / max(DENSE_DECAY_TAU_STEPS, 1e-6))

    if out_components is not None:
        out_components.clear()

    def _add_comp(key: str, val: float) -> None:
        if out_components is not None:
            out_components[key] = out_components.get(key, 0.0) + val

    prev_players = prev_obs["players"]
    curr_players = curr_obs["players"]
    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])

    if prev_alive == 1 and curr_alive == 0:
        _add_comp("agent_death", float(R_AGENT_DEATH))
        if out_components is not None:
            out_components["total"] = float(R_AGENT_DEATH)
        return float(R_AGENT_DEATH), episode_state

    reward = 0.0

    prev_enemies_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemies_alive = _enemy_alive_count(curr_players, agent_id)

    if curr_alive == 1 and curr_enemies_alive < prev_enemies_alive:
        v = R_ENEMY_DEATH * float(prev_enemies_alive - curr_enemies_alive)
        reward += v
        _add_comp("enemy_death", v)
    if (
        curr_alive == 1
        and curr_enemies_alive == 0
        and prev_enemies_alive > 0
    ):
        reward += R_LAST_ALIVE
        _add_comp("last_alive", float(R_LAST_ALIVE))

    prev_boxes = _box_count(prev_obs)
    curr_boxes = _box_count(curr_obs)
    if curr_boxes < prev_boxes:
        v = R_BLOCK_DESTROYED * float(prev_boxes - curr_boxes)
        reward += v
        _add_comp("block_destroyed", v)

    reward += R_TIME_STEP
    _add_comp("time_step", float(R_TIME_STEP))

    prev_x = int(prev_players[agent_id][0])
    prev_y = int(prev_players[agent_id][1])
    curr_x = int(curr_players[agent_id][0])
    curr_y = int(curr_players[agent_id][1])
    moved = prev_x != curr_x or prev_y != curr_y

    dense_pos_this_step = 0.0

    if curr_alive == 1 and moved:
        prev_d = _manhattan_to_nearest_alive_enemy(prev_players, agent_id, prev_x, prev_y)
        curr_d = _manhattan_to_nearest_alive_enemy(curr_players, agent_id, curr_x, curr_y)
        if prev_d is not None and curr_d is not None:
            improving_best = episode_state.best_dist is None or curr_d < episode_state.best_dist

            if curr_d < prev_d and improving_best:
                v = R_CLOSER_STEP * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("closer_step", v)
            elif curr_d > prev_d:
                v = R_FARTHER_STEP * decay
                reward += v
                _add_comp("farther_step", v)

            if improving_best:
                v = R_CLOSEST_SO_FAR * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("closest_so_far", v)
                episode_state.best_dist = float(curr_d)

    if curr_alive == 1:
        in_blast, _ = _blast_status_at(curr_obs, curr_x, curr_y)
        if in_blast:
            reward += R_IN_BLAST
            _add_comp("in_blast", float(R_IN_BLAST))
        elif _any_bombs(curr_obs):
            dbomb = _min_manhattan_to_bomb_center(curr_obs, curr_x, curr_y)
            if dbomb is not None and dbomb <= SAFE_NEAR_BOMB_RADIUS:
                v = R_SAFE_NEAR_BOMB * decay
                reward += v
                dense_pos_this_step += v
                _add_comp("safe_near_bomb", v)

    # Cap only positive dense shaping (not time / death / blocks / kills).
    cap_room = max(0.0, DENSE_SHAPING_CAP_PER_EPISODE - episode_state.dense_cumulative)
    if dense_pos_this_step > cap_room:
        scale = (cap_room / dense_pos_this_step) if dense_pos_this_step > 0 else 0.0
        if out_components is not None:
            for key in ("closer_step", "closest_so_far", "safe_near_bomb"):
                if key in out_components:
                    out_components[key] *= scale
        reward -= dense_pos_this_step * (1.0 - scale)
        dense_pos_this_step = cap_room

    episode_state.dense_cumulative += dense_pos_this_step
    if out_components is not None:
        out_components["dense_applied"] = dense_pos_this_step
        out_components["dense_decay"] = decay
        out_components["total"] = float(reward)

    return float(reward), episode_state


if __name__ == "__main__":
    es = EpisodeRewardState()

    # Death
    po = {"map": np.zeros((3, 3), dtype=np.int8), "players": np.array([[1, 1, 1, 1, 0]], dtype=np.int8), "bombs": np.zeros((0, 4), dtype=np.int8)}
    co = {"map": np.zeros((3, 3), dtype=np.int8), "players": np.array([[1, 1, 0, 1, 0]], dtype=np.int8), "bombs": np.zeros((0, 4), dtype=np.int8)}
    r, _ = compute_reward_icec(po, co, 0, es)
    assert abs(r - R_AGENT_DEATH) < 1e-9, r

    # Time step only (standing still, alive, no bombs)
    es2 = EpisodeRewardState()
    r, _ = compute_reward_icec(po, po, 0, es2)
    assert abs(r - R_TIME_STEP) < 1e-9, r

    # Movement + closer enemy (two players)
    m = np.zeros((5, 5), dtype=np.int8)
    po3 = {
        "map": m,
        "players": np.array([[1, 2, 1, 1, 0], [3, 2, 1, 1, 0]], dtype=np.int8),
        "bombs": np.zeros((0, 4), dtype=np.int8),
    }
    co3 = {
        "map": m,
        "players": np.array([[2, 2, 1, 1, 0], [3, 2, 1, 1, 0]], dtype=np.int8),
        "bombs": np.zeros((0, 4), dtype=np.int8),
    }
    es3 = EpisodeRewardState()
    r, _ = compute_reward_icec(po3, co3, 0, es3)
    decay1 = math.exp(-1.0 / DENSE_DECAY_TAU_STEPS)
    # time + decayed closer + decayed closest-so-far (step 1)
    want = R_TIME_STEP + (R_CLOSER_STEP + R_CLOSEST_SO_FAR) * decay1
    assert abs(r - want) < 1e-9, (r, want)

    # Oscillation: after reaching best distance, stepping closer again without a new record gives no dense movement bonus.
    es4 = EpisodeRewardState()
    _, es4 = compute_reward_icec(po3, co3, 0, es4)
    assert es4.best_dist == 1.0
    r_b, es4 = compute_reward_icec(co3, po3, 0, es4)
    decay2 = math.exp(-2.0 / DENSE_DECAY_TAU_STEPS)
    want_b = R_TIME_STEP + R_FARTHER_STEP * decay2
    assert abs(r_b - want_b) < 1e-9, (r_b, want_b)
    r_c, es4 = compute_reward_icec(po3, co3, 0, es4)
    # Closer again to dist=1 but not strictly better than best_dist=1 → only time penalty
    assert abs(r_c - R_TIME_STEP) < 1e-9, (r_c, es4.best_dist)

    print("reward_02 self-tests passed.")
