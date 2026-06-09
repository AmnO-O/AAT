"""
train_scorer.py — Neural Scorer Training Pipeline  v2
==================================================
WHAT CHANGED vs v1 (and why)
─────────────────────────────

[FIX 1 — CRITICAL] Counterfactual 1-step rollouts replace shared return_30.

  The v1 bug: all candidates at step t share the SAME return_30 because only
  the rule agent's chosen action actually executed.  The network had no signal
  to discriminate between candidates; MSE ~0.935 with random baseline ~1.0
  means only 6.5% of target variance was explained.

  The fix: for each candidate action, we do a lightweight 1-step counterfactual:
    • execute the candidate in a *copy* of the env state (or compute it
      analytically via the shaped reward function for moves, which is fast)
    • label = immediate_shaped_reward(candidate) + GAMMA * V_rule(next_state)
  where V_rule(next_state) is approximated by the return_30 from the rule
  agent continuing from there (which we already have from the forward pass).

  Practically this means:
    label(candidate_a) = step_reward(if I took a) + gamma * return_from_t+1
  For MOVE actions this is analytically cheap.
  For PLACE_BOMB we use the bomb_value_full score as a proxy since the env
  step that follows a bomb placement is essentially identical to STOP.

  This gives each candidate a DIFFERENT label, which is the actual signal.

[FIX 2] Pairwise ranking loss (Bradley-Terry) replaces MSE regression.

  MSE penalizes prediction magnitude; ranking penalizes wrong ordering.
  For a candidate scorer, we only care about which of two candidates is better.
  Loss = mean over pairs (i,j) of: BCE(sigmoid(score_i - score_j), label_ij)
  where label_ij = 1 if return_i > return_j, 0.5 if equal.

  This loss is much less sensitive to the absolute scale of noisy returns
  and directly trains the argmax selection used at inference.

[FIX 3] Candidate features: 3 new features added (CAND_DIM 7 → 10).

  New features encode destination-conditioned information:
    f8: boxes in blast radius from destination (proxy for future bomb value there)
    f9: escape count from destination (how many safe exits after moving there)
    f10: distance to nearest enemy from destination (closer = more combat value)

  These are action-specific and not in the state context, so they help the
  net distinguish movement candidates that look identical on f1-f7.

  NOTE: submission_v1.py must be updated to match CAND_DIM=10 at inference.
  A helper patch is printed at the end of training.

[FIX 4] Agent ID shuffle in evaluation.

  v1 always evaluates hybrid as agent 0 (top-left corner).
  Different corners have different early-game item/box patterns.
  Now cid rotates: gi % 4, same as the training collection.

[NOT CHANGED] Network size 34→64→32→1.

  With the new counterfactual labels, this architecture should extract much
  more signal. We increase to 34+3=37 input → 64 → 32 → 1 (same depth).
  A larger net is an option if val loss stalls after these fixes are applied.
"""

import argparse
import os
import sys
import copy
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())

# ── Import game engine ────────────────────────────────────────────────────────
from engine.game import BomberEnv   # adjust path if needed

# ── Import the rule agent internals ──────────────────────────────────────────
# ── Import các đối thủ đa dạng từ pool ────────────────────────────────────────
from agent.submission_v1.submission_v1 import Agent as MyTrainingAgent # Agent chính bạn muốn train

from agent.smarter_rule_agent import SmarterRuleAgent as SmarterRuleAgent
from agent.tactical_rule_agent import TacticalRuleAgent as TacticalRuleAgent
from agent.genius_rule_agent import GeniusRuleAgent as GeniusRuleAgent
from agent.samnu_agent import Agent as SamnuAgent

from agent.submission_v1.submission_v1 import (
    Agent as RuleAgent,
    _build_danger_timed,
    _encode_state_context,
    _encode_candidate_features,
    _generate_candidates,
    _bomb_value_full,
    _count_nearby_armed,
    _enemy_predicted_blast,
    _walkable, _blast_tiles, _reachable_safe_count,
    STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB,
    MOVE_ACTIONS, MOVES,
    BOARD_CX, BOARD_CY,
    MAX_RADIUS, MAX_CAPACITY, BOMB_TIMER,
    GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY,
    STATE_DIM, CAND_DIM, TOTAL_IN_DIM,
    TIME_BUDGET_S,
)

# v2: extended candidate feature dimension
CAND_DIM_V2   = CAND_DIM + 3   # 7 + 3 = 10 new features
TOTAL_IN_DIM_V2 = STATE_DIM + CAND_DIM_V2  # 27 + 10 = 37

# ── Configuration ─────────────────────────────────────────────────────────────
N_GAMES          = 5_000    # self-play games for data collection
N_EVAL_GAMES     = 200      # games for eval (hybrid vs rule)
RETURN_STEPS     = 30       # horizon for return computation
GAMMA            = 0.97     # discount factor
TRAIN_EPOCHS     = 40
BATCH_SIZE       = 512
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
VAL_FRAC         = 0.10     # fraction held out for validation
MAX_STEPS        = 500      # max game steps
DATASET_PATH     = "scorer_dataset_v2.npz"
SCORER_PATH      = "./agent/submission_v1/scorer.pt"
BEST_PATH        = "./agent/submission_v1/scorer_best.pt"
SEED             = 42

# v2 training options
USE_COUNTERFACTUAL_LABELS = True   # Fix 1: per-candidate labels instead of shared return
USE_RANKING_LOSS          = True   # Fix 2: pairwise Bradley-Terry loss instead of MSE
RANKING_MARGIN            = 0.05   # minimum return difference to count as a preference

# Fixed map seeds: training / eval split
N_TRAIN_MAPS     = 200
N_EVAL_MAPS      = 50
_TRAIN_SEEDS     = [300_000 + i*137 for i in range(N_TRAIN_MAPS)]
_EVAL_SEEDS      = [900_000 + i*137 for i in range(N_EVAL_MAPS)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# =============================================================================
# ScorerNet v2  (wider input: 37 = 27 state + 10 candidate features)
# =============================================================================

class ScorerNetV2(nn.Module):
    """
    37 → 64 → 32 → 1.
    Same depth as v1 ScorerNet but wider input to accommodate 3 new candidate
    features.  Copy this class into submission_v1.py when deploying.
    """
    def __init__(self, in_dim: int = TOTAL_IN_DIM_V2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =============================================================================
# Extended candidate feature encoder  (CAND_DIM_V2 = 10)
# =============================================================================

def _encode_candidate_features_v2(action, my_r, my_c, my_radius,
                                   grid, players, agent_id,
                                   danger_any, bomb_pos, danger_by_time,
                                   obs_bombs):
    """
    10 floats = original 7 (from submission_v1) + 3 new action-specific features.

    New features (indices 7-9):
      f8  boxes_from_dest: boxes in blast radius of a default bomb placed at
          destination. Zero for STOP. For movement, this previews 'if I move
          here and then bomb, how much board control do I gain?'

      f9  escape_exits_from_dest: number of walkable non-danger neighbours from
          destination (proxy for how trapped the agent becomes after this move).
          Normalised by 4.

      f10 enemy_dist_from_dest: normalised BFS distance to nearest live enemy
          from destination. Closer = higher combat relevance.
    """
    # Base 7 features from v1
    base = list(_encode_candidate_features(
        action, my_r, my_c, my_radius,
        grid, players, agent_id,
        danger_any, bomb_pos, danger_by_time,
        obs_bombs
    ))
    # Drop the trailing 0.0 padding (last element of base is always 0.0 spare)
    # and extend with 3 new features.

    h, w = grid.shape

    # Destination position
    if action == STOP or action == PLACE_BOMB:
        nr, nc = my_r, my_c
    else:
        dr, dc = MOVES[action]
        nr, nc = my_r + dr, my_c + dc

    # f8: boxes in blast radius from destination (using agent's current radius)
    if action == PLACE_BOMB:
        # Already captured in bval_n (feature 4); use 0 to avoid double-encoding
        boxes_from_dest = 0.0
    else:
        dest_blast = _blast_tiles(nr, nc, my_radius, grid)
        boxes_from_dest = min(
            sum(1 for r, c in dest_blast if grid[r, c] == BOX), 8
        ) / 8.0

    # f9: open non-danger neighbours at destination (escape richness)
    open_safe_at_dest = sum(
        1 for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if _walkable(nr + dr2, nc + dc2, grid, bomb_pos)
        and (nr + dr2, nc + dc2) not in danger_any
    ) / 4.0

    # f10: BFS distance to nearest enemy from destination (normalised)
    live_enemies = {
        (int(players[i][0]), int(players[i][1]))
        for i in range(len(players))
        if i != agent_id and int(players[i][2]) == 1
    }
    if not live_enemies:
        enemy_dist_dest = 1.0
    else:
        vis = {(nr, nc)}
        q   = deque([((nr, nc), 0)])
        enemy_dist_dest = 1.0
        while q:
            pos2, d = q.popleft()
            if pos2 in live_enemies:
                enemy_dist_dest = d / 16.0
                break
            if d >= 16:
                continue
            for dr3, dc3 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                np2 = (pos2[0] + dr3, pos2[1] + dc3)
                if np2 not in vis and _walkable(np2[0], np2[1], grid, bomb_pos):
                    vis.add(np2)
                    q.append((np2, d + 1))

    return base + [boxes_from_dest, open_safe_at_dest, enemy_dist_dest]


# =============================================================================
# Counterfactual label computation
# =============================================================================

def _counterfactual_label(action, my_r, my_c, my_radius, bombs_left,
                           grid, players, agent_id,
                           danger_any, bomb_pos, danger_by_time,
                           obs_bombs, future_return: float) -> float:
    """
    Compute per-candidate label: immediate_reward(action) + gamma * future_return.

    For movement actions the immediate reward is computed analytically:
      - reaching an item tile: +item_reward
      - entering danger: -2.0 (heavy penalty for walking into blast)
      - safe move: small positive (0.0005 survival proxy)

    For PLACE_BOMB the immediate reward is approximated from bomb_value_full
    (the rule layer already computed this).

    For STOP: small survival tick.

    The future_return is the game's actual discounted return from t+1 onward
    (same for all candidates at this step, but now it only covers steps 2-30,
    not step 1, so it acts as a shared baseline rather than the whole signal).

    This gives DIFFERENT labels per candidate because the immediate reward
    component differs, making the ranking task well-defined.
    """
    h, w = grid.shape
    imm = 0.0

    if action == STOP:
        nr, nc = my_r, my_c
        imm += 0.0005  # survival tick
        if (nr, nc) in danger_any:
            imm -= 1.5   # staying in danger is bad

    elif action == PLACE_BOMB:
        if bombs_left > 0 and (my_r, my_c) not in bomb_pos:
            bval, hits_box, hits_enemy = _bomb_value_full(
                my_r, my_c, my_radius, grid, players, agent_id,
                danger_any, obs_bombs, bomb_pos, danger_by_time
            )
            # positive immediate value = boxes destroyed + enemy pressure
            imm += min(bval, 10.0) * 0.15
            # penalty if can't escape (rule layer should have already filtered,
            # but if it slipped through, penalise)
        else:
            imm -= 0.5  # illegal bomb attempt

    else:
        # Movement action
        dr, dc = MOVES[action]
        nr, nc = my_r + dr, my_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            imm -= 3.0   # off-board (shouldn't happen after rule filter)
        elif grid[nr, nc] in (WALL, BOX) or (nr, nc) in bomb_pos:
            imm -= 3.0   # blocked move
        else:
            imm += 0.0005   # survival
            # item collection bonus
            if grid[nr, nc] == ITEM_RADIUS:
                imm += 0.06
            elif grid[nr, nc] == ITEM_CAPACITY:
                imm += 0.08
            # danger entry penalty
            if (nr, nc) in danger_any:
                imm -= 2.0
            elif (nr, nc) in danger_by_time.get(2, set()):
                imm -= 0.5   # softer penalty: entering t=2 danger zone

    return float(imm + GAMMA * future_return)


# =============================================================================
# Pairwise ranking loss (Bradley-Terry)
# =============================================================================

def ranking_loss_fn(scores: torch.Tensor, labels: torch.Tensor,
                    margin: float = RANKING_MARGIN) -> torch.Tensor:
    """
    Pairwise Bradley-Terry ranking loss over a batch of (state, candidate) pairs.

    Inputs:
      scores : (N,)  — model predictions
      labels : (N,)  — counterfactual return labels
    
    For every pair (i, j) where |label_i - label_j| > margin:
      target = 1.0 if label_i > label_j else 0.0
      loss   = BCE(sigmoid(scores_i - scores_j), target)

    We use a vectorised O(N^2) computation; for typical batch sizes (512)
    this is fast enough on CPU.
    """
    # Pairwise label differences: (N, N)
    diff = labels.unsqueeze(0) - labels.unsqueeze(1)   # diff[i,j] = label_i - label_j
    mask = diff.abs() > margin                          # only pairs with meaningful difference

    if mask.sum() == 0:
        # Fallback to MSE when no pairs are distinguishable (degenerate batch)
        return torch.nn.functional.mse_loss(scores, labels)

    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (N, N)
    targets    = (diff > 0).float()                          # 1.0 where i is better

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        score_diff[mask], targets[mask], reduction="mean"
    )
    return bce


# =============================================================================
# Reward shaping  (unchanged from v1)
# =============================================================================

def _step_reward(prev_obs, next_obs, my_id, action, terminated, truncated):
    """
    Lightweight reward for return computation during data collection.
    Mirrors the signals the agent actually cares about.
    """
    r  = 0.0
    pp = prev_obs["players"]; np_ = next_obs["players"]
    pm = prev_obs["map"];     nm  = next_obs["map"]

    if my_id < len(pp) and my_id < len(np_):
        pa,na = int(pp[my_id][2]),int(np_[my_id][2])
        if pa==1 and na==0: r -= 4.0          # death
        if pa==1 and na==1: r += 0.0005       # survival tick

    # kills
    if my_id<len(pp) and my_id<len(np_):
        pe = sum(int(pp[i][2]) for i in range(len(pp)) if i!=my_id)
        ne = sum(int(np_[i][2]) for i in range(len(np_)) if i!=my_id)
        kills = max(0,pe-ne)
        if kills>0: r += (3.0 if ne==0 else 2.0)*kills

    # boxes
    boxes = max(0, int(np.sum(pm==2))-int(np.sum(nm==2)))
    r += 0.04*boxes

    # items
    if my_id<len(pp) and my_id<len(np_) and int(pp[my_id][2])==1 and int(np_[my_id][2])==1:
        if int(np_[my_id][4])>int(pp[my_id][4]): r+=0.06

    # terminal
    if terminated or truncated:
        if my_id<len(np_) and int(np_[my_id][2])==1:
            r += 10.0 if sum(int(np_[i][2]) for i in range(len(np_)))==1 else 0.5

    return float(np.clip(r,-12.0,15.0))


# =============================================================================
# Data collection
# =============================================================================

def collect_dataset(n_games=N_GAMES, verbose=True):
    """
    Run n_games self-play games.

    v2 changes:
      - Records extended 10-dim candidate features (CAND_DIM_V2).
      - When USE_COUNTERFACTUAL_LABELS=True: labels each candidate with
        _counterfactual_label(action) + gamma * future_return_from_t+1.
        This gives every candidate at a step a DIFFERENT label, which is the
        signal the network needs to discriminate between them.
      - When False: falls back to v1 behaviour (shared return_30) for comparison.
    """
    all_states  = []
    all_cands   = []
    all_returns = []

    for gi in range(n_games):
        map_seed = _TRAIN_SEEDS[gi % N_TRAIN_MAPS]
        env      = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs      = env.reset()

        agents = [MyTrainingAgent(0)]
        for i in range(1, 4):
            pool_choice = random.choice([
                "smarter", "tactical", "genius", "samnu", "myself"
            ])
            if pool_choice == "smarter":
                agents.append(SmarterRuleAgent(i))
            elif pool_choice == "tactical":
                agents.append(TacticalRuleAgent(i))
            elif pool_choice == "genius":
                agents.append(GeniusRuleAgent(i))
            elif pool_choice == "samnu":
                agents.append(SamnuAgent(i))
            else:
                agents.append(MyTrainingAgent(i))

        # Per-agent episode buffers: (state_ctx, [(action, cand_feat)], step_reward)
        ep_bufs  = [[] for _ in range(4)]
        step     = 0
        done     = False

        while not done:
            actions   = [0, 0, 0, 0]
            step_data = [None] * 4

            for cid in range(4):
                if int(obs["players"][cid][2]) != 1:
                    continue

                me           = obs["players"][cid]
                my_r, my_c   = int(me[0]), int(me[1])
                bombs_left   = int(me[3])
                radius_bonus = int(me[4])
                my_radius    = max(1, min(MAX_RADIUS, 1 + radius_bonus))
                pos          = (my_r, my_c)

                danger_by_time, danger_any = _build_danger_timed(obs)
                bomb_pos     = {(int(b[0]), int(b[1])) for b in obs["bombs"]}
                nearby_armed = _count_nearby_armed(my_r, my_c, obs["players"], cid)
                combat_mode  = nearby_armed >= 1
                extra_danger = set()
                if combat_mode:
                    extra_danger = _enemy_predicted_blast(
                        my_r, my_c, obs["players"], cid, obs["map"])

                t0_fake = time.perf_counter() - 0.001
                cands = _generate_candidates(
                    pos, my_r, my_c, my_radius, bombs_left,
                    obs["map"], obs["players"], cid,
                    danger_any, bomb_pos, danger_by_time,
                    extra_danger, combat_mode, nearby_armed,
                    obs["bombs"], t0_fake
                )

                try:
                    rule_a = agents[cid].act(obs)
                except:
                    rule_a = STOP

                if len(cands) >= 2:
                    pos_hist = getattr(agents[cid], 'pos_history', [])
                    is_stuck = (
                        len(pos_hist) >= 16
                        and len(set(pos_hist)) <= 3
                        and len(obs["bombs"]) == 0
                        and pos not in danger_any
                    )
                    ctx = _encode_state_context(
                        my_r, my_c, my_radius, bombs_left,
                        obs["map"], obs["players"], cid,
                        danger_any, bomb_pos, danger_by_time, step
                    )
                    ctx[20] = float(is_stuck)
                    ctx[21] = float(combat_mode)

                    # v2: use extended 10-dim candidate features
                    cand_feats = []
                    for a in cands:
                        cf = _encode_candidate_features_v2(
                            a, my_r, my_c, my_radius,
                            obs["map"], obs["players"], cid,
                            danger_any, bomb_pos, danger_by_time,
                            obs["bombs"]
                        )
                        cand_feats.append((a, cf))

                    step_data[cid] = (ctx, cand_feats, my_r, my_c, my_radius,
                                      bombs_left, danger_any, bomb_pos,
                                      danger_by_time, obs["bombs"])

                actions[cid] = rule_a

            prev_obs = obs
            obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            for cid in range(4):
                rew = _step_reward(prev_obs, obs, cid, actions[cid], terminated, truncated)
                if step_data[cid] is not None:
                    (ctx, cand_feats, my_r, my_c, my_radius, bombs_left,
                     danger_any, bomb_pos, danger_by_time, obs_bombs) = step_data[cid]
                    ep_bufs[cid].append((ctx, cand_feats, rew,
                                         my_r, my_c, my_radius, bombs_left,
                                         danger_any, bomb_pos, danger_by_time, obs_bombs))

        # ── Compute labels per candidate ───────────────────────────────────────
        for cid in range(4):
            buf = ep_bufs[cid]
            T   = len(buf)
            for t in range(T):
                # future_return: discounted sum from t+1 onward (shared baseline)
                future_ret = 0.0
                g = 1.0
                for k in range(1, RETURN_STEPS + 1):
                    if t + k >= T:
                        break
                    future_ret += g * buf[t + k][2]
                    g *= GAMMA

                (ctx, cand_feats, _, my_r, my_c, my_radius, bombs_left,
                 danger_any, bomb_pos, danger_by_time, obs_bombs) = buf[t]

                for a, cf in cand_feats:
                    all_states.append(ctx)
                    all_cands.append(cf)

                    if USE_COUNTERFACTUAL_LABELS:
                        # v2: per-candidate label = immediate_reward(a) + gamma * future
                        label = _counterfactual_label(
                            a, my_r, my_c, my_radius, bombs_left,
                            obs["map"], obs["players"], cid,
                            danger_any, bomb_pos, danger_by_time,
                            obs_bombs, future_ret
                        )
                    else:
                        # v1 fallback: all candidates get the same shared return
                        label = future_ret

                    all_returns.append(label)

        if verbose and (gi + 1) % 500 == 0:
            print(f"  collected game {gi+1}/{n_games} | "
                  f"samples so far: {len(all_returns):,}")

    states_arr  = np.array(all_states,  dtype=np.float32)
    cands_arr   = np.array(all_cands,   dtype=np.float32)
    returns_arr = np.array(all_returns, dtype=np.float32)

    print(f"\nDataset: {len(returns_arr):,} samples")
    print(f"  return range  [{returns_arr.min():.3f}, {returns_arr.max():.3f}]")
    print(f"  return mean   {returns_arr.mean():.4f}  std {returns_arr.std():.4f}")

    # Quick sanity: check that at the same step, candidates now have different labels
    if USE_COUNTERFACTUAL_LABELS and len(returns_arr) > 10:
        # A step with >=2 candidates will have them in consecutive positions
        # (they were appended together in the inner loop)
        # Just report the std of the first 1000 to check variance
        sample_std = returns_arr[:min(1000, len(returns_arr))].std()
        print(f"  label std (first 1000): {sample_std:.4f}  "
              f"(v1 ~0.3, v2 should be higher if counterfactual labels are working)")

    return states_arr, cands_arr, returns_arr





def save_dataset(states, cands, returns, path=DATASET_PATH):
    np.savez_compressed(path, states=states, cands=cands, returns=returns)
    print(f"Saved dataset → {path}  ({os.path.getsize(path)//1024} KB)")


def load_dataset(path=DATASET_PATH):
    d = np.load(path)
    return d["states"], d["cands"], d["returns"]


# =============================================================================
# Training
# =============================================================================

def normalise_returns(returns):
    """
    Normalise per-game is already done implicitly via the return horizon.
    Here we do a global z-score to keep loss magnitudes sensible.
    """
    mu,sigma = returns.mean(), returns.std()+1e-8
    return (returns-mu)/sigma, mu, sigma


def build_dataloaders(states, cands, returns):
    """Split into train/val, return DataLoaders."""
    N = len(returns)
    idx = np.random.permutation(N)
    n_val = max(1, int(N*VAL_FRAC))
    val_idx,trn_idx = idx[:n_val],idx[n_val:]

    def _dl(idx_, shuffle):
        X = torch.tensor(
            np.concatenate([states[idx_],cands[idx_]],axis=1),
            dtype=torch.float32)
        Y = torch.tensor(returns[idx_], dtype=torch.float32)
        ds = TensorDataset(X,Y)
        return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=shuffle,
                          num_workers=0,pin_memory=torch.cuda.is_available())

    return _dl(trn_idx,True), _dl(val_idx,False)


def train(states, cands, returns, verbose=True):
    """
    Train ScorerNetV2.

    v2 changes:
      - Uses ScorerNetV2 (input dim = STATE_DIM + CAND_DIM_V2 = 37).
      - When USE_RANKING_LOSS=True: uses pairwise Bradley-Terry loss.
        Falls back to MSE if USE_RANKING_LOSS=False (for comparison).
    """
    ret_norm, mu, sigma = normalise_returns(returns)
    trn_dl, val_dl      = build_dataloaders(states, cands, ret_norm)

    model     = ScorerNetV2(in_dim=TOTAL_IN_DIM_V2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    mse_fn    = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nScorerNetV2: {n_params:,} parameters  device={DEVICE}")
    print(f"  Input dim: {TOTAL_IN_DIM_V2} (state {STATE_DIM} + cand {CAND_DIM_V2})")
    print(f"  Loss: {'Pairwise ranking (Bradley-Terry)' if USE_RANKING_LOSS else 'MSE'}")
    print(f"Train samples: {len(trn_dl.dataset):,}  "
          f"Val samples: {len(val_dl.dataset):,}\n")

    best_val = float("inf")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        # train
        model.train(); trn_loss = 0.0; nb = 0
        for X, Y in trn_dl:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            if USE_RANKING_LOSS:
                loss = ranking_loss_fn(pred, Y)
            else:
                loss = mse_fn(pred, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            trn_loss += loss.item(); nb += 1
        trn_loss /= max(nb, 1)

        # validate — always report MSE as well for interpretability
        model.eval(); val_loss = 0.0; val_mse = 0.0; nb = 0
        with torch.no_grad():
            for X, Y in val_dl:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                pred = model(X)
                if USE_RANKING_LOSS:
                    val_loss += ranking_loss_fn(pred, Y).item()
                else:
                    val_loss += mse_fn(pred, Y).item()
                val_mse += mse_fn(pred, Y).item()
                nb += 1
        val_loss /= max(nb, 1)
        val_mse  /= max(nb, 1)
        scheduler.step()

        if verbose:
            loss_label = "rank" if USE_RANKING_LOSS else "mse"
            print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS} | "
                  f"trn={trn_loss:.4f}  val_{loss_label}={val_loss:.4f}  "
                  f"val_mse={val_mse:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            if verbose:
                print(f"  ★ new best val={best_val:.4f} → {BEST_PATH}")

    # load best for final export
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    model.eval()
    torch.save(model.state_dict(), SCORER_PATH)
    print(f"\nSaved trained scorer → {SCORER_PATH}")
    print(f"Best val loss: {best_val:.4f}")
    return model, mu, sigma


# =============================================================================
# Evaluation: hybrid (rule+net) vs pure rule agent
# =============================================================================

class HybridAgent:
    """Eval wrapper: uses the rule agent but with neural scorer override."""
    def __init__(self, agent_id, scorer):
        from agent.submission_v1.submission_v1 import Agent as RA
        self._rule = RA(agent_id)
        self._rule._scorer = scorer
        self.agent_id = agent_id

    def act(self, obs):
        return self._rule.act(obs)


def evaluate(scorer, n_games=N_EVAL_GAMES, verbose=True):
    """
    Run n_games where one agent is hybrid, agents 1-3 are pure rule.

    v2: cid rotates (gi % 4) so the hybrid is tested from all 4 starting
    corners instead of always agent 0 (top-left).  This gives a fairer
    estimate of the scorer's value across different map positions.
    """
    scorer.eval()
    wins = draws = losses = kills_t = boxes_t = 0

    for gi in range(n_games):
        ms  = _EVAL_SEEDS[gi % N_EVAL_MAPS]
        cid = gi % 4   # v2: rotate across corners
        env = BomberEnv(max_steps=MAX_STEPS, seed=ms)
        obs = env.reset()
        init_boxes = int(np.sum(obs["map"] == 2))

        hybrid_ag = HybridAgent(cid, scorer)
        rule_ags  = {i: RuleAgent(i) for i in range(4) if i != cid}

        done = False; step = 0; kills = 0
        while not done:
            if int(obs["players"][cid][2]) != 1:
                break
            pe = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            acts = [0, 0, 0, 0]
            acts[cid] = hybrid_ag.act(obs)
            for pid, ag in rule_ags.items():
                acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            kills += max(0, pe - sum(int(obs["players"][i][2]) for i in range(4) if i != cid))
            done = bool(terminated or truncated); step += 1

        alive = [int(p[2]) for p in obs["players"]]
        boxes_destroyed = init_boxes - int(np.sum(obs["map"] == 2))
        if alive[cid] == 1 and sum(alive) == 1:
            wins += 1
        elif alive[cid] == 1:
            draws += 1
        else:
            losses += 1
        kills_t += kills; boxes_t += boxes_destroyed

    ng = max(1, n_games)
    wr = wins / ng; dr = draws / ng
    print(f"\nEval ({n_games}g) | W={wins} D={draws} L={losses} | "
          f"WR={wr:.2%} | kills={kills_t/ng:.2f} boxes={boxes_t/ng:.0f}")
    return wr


# =============================================================================
# Iterative refinement  (optional)
# =============================================================================

def iterative_train(rounds=3, games_per_round=2000):
    """
    Multi-round refinement.  Same structure as v1 but uses ScorerNetV2.
    """
    combined_states = np.zeros((0, STATE_DIM),    dtype=np.float32)
    combined_cands  = np.zeros((0, CAND_DIM_V2),  dtype=np.float32)
    combined_rets   = np.zeros((0,),               dtype=np.float32)

    best_wr = -1.0

    for rnd in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{rounds}")
        print(f"{'='*60}")

        print(f"\n[{rnd}] Collecting {games_per_round} games...")
        s, c, r = collect_dataset(n_games=games_per_round)

        combined_states = np.concatenate([combined_states, s])
        combined_cands  = np.concatenate([combined_cands,  c])
        combined_rets   = np.concatenate([combined_rets,   r])
        print(f"[{rnd}] Combined dataset: {len(combined_rets):,} samples")

        print(f"\n[{rnd}] Training...")
        scorer, _, _ = train(combined_states, combined_cands, combined_rets)

        print(f"\n[{rnd}] Evaluating...")
        wr = evaluate(scorer)

        if wr > best_wr:
            best_wr = wr
            torch.save(scorer.state_dict(), BEST_PATH)
            print(f"[{rnd}] ★ New best WR={wr:.2%} → {BEST_PATH}")

    best = ScorerNetV2(in_dim=TOTAL_IN_DIM_V2).to(DEVICE)
    best.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
    best.eval()
    torch.save(best.state_dict(), SCORER_PATH)
    print(f"\nFinal scorer exported → {SCORER_PATH}")
    _print_deployment_note()
    return best


def _print_deployment_note():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  DEPLOYMENT: update submission_v1.py with these changes             ║
╠══════════════════════════════════════════════════════════════════════╣
║  1. Copy ScorerNetV2 class into submission_v1.py (replace ScorerNet)║
║  2. Copy _encode_candidate_features_v2 function                     ║
║  3. Update: CAND_DIM = 10   (was 7)                                 ║
║             TOTAL_IN_DIM = 37  (was 34)                             ║
║  4. In _score_candidates_with_net: call _encode_candidate_features_v2║
║     instead of _encode_candidate_features                           ║
║  5. In Agent.__init__: replace ScorerNet() with ScorerNetV2()       ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ScorerNetV2 for hybrid Bomberland agent")
    parser.add_argument("--n-games",       type=int,  default= 500,
                        help="self-play games to collect")
    parser.add_argument("--epochs",        type=int,  default=TRAIN_EPOCHS,
                        help="training epochs")
    parser.add_argument("--rounds",        type=int,  default=3,
                        help="iterative refinement rounds (1 = single pass)")
    parser.add_argument("--skip-collect",  action="store_true",
                        help="skip data collection, load existing dataset")
    parser.add_argument("--eval-only",     action="store_true",
                        help="only run evaluation with existing scorer.pt")
    parser.add_argument("--n-eval",        type=int,  default=N_EVAL_GAMES)
    parser.add_argument("--no-counterfactual", action="store_true",
                        help="disable counterfactual labels (v1 behaviour)")
    parser.add_argument("--no-ranking",    action="store_true",
                        help="disable ranking loss, use MSE (v1 behaviour)")
    args = parser.parse_args()

    # Apply CLI overrides
    global USE_COUNTERFACTUAL_LABELS, USE_RANKING_LOSS
    if args.no_counterfactual:
        USE_COUNTERFACTUAL_LABELS = False
    if args.no_ranking:
        USE_RANKING_LOSS = False

    print(f"Device: {DEVICE}")
    print(f"Config: {args.n_games} games | {args.epochs} epochs | "
          f"{args.rounds} rounds")
    print(f"  counterfactual_labels={USE_COUNTERFACTUAL_LABELS}  "
          f"ranking_loss={USE_RANKING_LOSS}\n")

    # ── eval only ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(SCORER_PATH):
            print(f"ERROR: {SCORER_PATH} not found."); return
        scorer = ScorerNetV2(in_dim=TOTAL_IN_DIM_V2).to(DEVICE)
        scorer.load_state_dict(torch.load(SCORER_PATH, map_location=DEVICE))
        scorer.eval()
        evaluate(scorer, n_games=args.n_eval)
        return

    # ── multi-round iterative ─────────────────────────────────────────────
    if args.rounds > 1:
        iterative_train(rounds=args.rounds,
                        games_per_round=args.n_games // args.rounds)
        return

    # ── single-pass ───────────────────────────────────────────────────────
    if args.skip_collect and os.path.exists(DATASET_PATH):
        print(f"Loading existing dataset from {DATASET_PATH}...")
        states, cands, returns = load_dataset()
    else:
        print(f"Collecting {args.n_games} self-play games...")
        states, cands, returns = collect_dataset(n_games=args.n_games)
        save_dataset(states, cands, returns)

    print(f"\nTraining for {args.epochs} epochs...")
    scorer, _, _ = train(states, cands, returns)

    print(f"\nEvaluating hybrid vs pure rule ({args.n_eval} games)...")
    evaluate(scorer, n_games=args.n_eval)

    _print_deployment_note()
    print("\nDone. To use the scorer, place scorer.pt next to agent.py.")


if __name__ == "__main__":
    main()