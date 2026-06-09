"""
train_scorer.py — Neural Scorer Training Pipeline
==================================================
Trains the ScorerNet that sits inside the hybrid agent.

WHAT IT DOES
────────────
1. SELF-PLAY DATA COLLECTION
   Runs N_GAMES games with the pure rule agent playing all 4 positions.
   For every step where the agent has >=2 rule-safe candidates, records:
     (state_context, candidate_action, return_30)
   where return_30 is the discounted sum of rewards over the next RETURN_STEPS
   steps (or until death/game-end).

2. DATASET BUILDING
   Normalises returns per-game, saves to scorer_dataset.npz.

3. TRAINING
   Trains ScorerNet (34→64→32→1) with MSE loss to predict return_30 for
   each (state+candidate) pair. This is purely supervised — no PPO, no
   rollout tricks. Run time: ~10 min on CPU for 5000 games.

4. EVALUATION
   Runs eval games: hybrid (rule+net) vs pure rule agent, reports win rate.

5. EXPORT
   Saves scorer.pt — the file agent.py loads at startup.

USAGE
─────
  # Basic run (pure self-play, 5000 games):
  python train_scorer.py

  # Resume from existing dataset (skip collection):
  python train_scorer.py --skip-collect

  # Evaluate only:
  python train_scorer.py --eval-only

  # Custom game count:
  python train_scorer.py --n-games 10000

REQUIREMENTS
────────────
  pip install torch numpy tqdm
  The engine (engine/game.py) and agent baselines must be importable.
  The rule agent (agent.py) must be in the same directory.

OUTPUT FILES
────────────
  scorer_dataset.npz   raw training data
  scorer.pt            trained model weights (load this in agent.py)
  scorer_best.pt       best checkpoint by eval win rate
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
    STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB,
    MOVE_ACTIONS, MOVES,
    BOARD_CX, BOARD_CY,
    MAX_RADIUS, MAX_CAPACITY, BOMB_TIMER,
    GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY,
    STATE_DIM, CAND_DIM, TOTAL_IN_DIM,
    ScorerNet,
    TIME_BUDGET_S,
)

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
DATASET_PATH     = "scorer_dataset.npz"
SCORER_PATH      = "./agent/submission_v1/scorer.pt"
BEST_PATH        = "./agent/submission_v1/scorer_best.pt"
SEED             = 42

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
# Reward shaping  (simple, quick to compute)
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
    Run n_games self-play games. At each step, for each agent:
      - compute rule-safe candidates
      - record (state_ctx, candidate_action) for ALL candidates
      - after RETURN_STEPS steps, label each with discounted return
    Returns arrays: states (N,STATE_DIM), cands (N,CAND_DIM), returns (N,)
    """
    all_states  = []
    all_cands   = []
    all_returns = []

    # We run 4 rule agents simultaneously
    for gi in range(n_games):
        map_seed = _TRAIN_SEEDS[gi % N_TRAIN_MAPS]
        env      = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs      = env.reset()
        
        agents = [MyTrainingAgent(0)]
        
        for i in range(1, 4):
            pool_choice = random.choice([
                "smarter", 
                "tactical", 
                "genius", 
                "samnu",
                "myself" # Cho đấu với chính bản thân mình luôn
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

        # Per-agent episode buffers: list of (state_ctx, cand_feat, step_reward)
        ep_bufs  = [[] for _ in range(4)]
        step     = 0
        done     = False

        while not done:
            actions  = [0,0,0,0]
            step_data = [None]*4   # (state_ctx, candidate_feats_list) per agent

            for cid in range(4):
                if int(obs["players"][cid][2]) != 1:
                    continue

                me          = obs["players"][cid]
                my_r,my_c   = int(me[0]),int(me[1])
                bombs_left  = int(me[3])
                radius_bonus= int(me[4])
                my_radius   = max(1,min(MAX_RADIUS,1+radius_bonus))
                pos         = (my_r,my_c)

                danger_by_time,danger_any = _build_danger_timed(obs)
                bomb_pos     = {(int(b[0]),int(b[1])) for b in obs["bombs"]}
                nearby_armed = _count_nearby_armed(my_r,my_c,obs["players"],cid)
                combat_mode  = nearby_armed >= 1
                extra_danger = set()
                if combat_mode:
                    extra_danger = _enemy_predicted_blast(
                        my_r,my_c,obs["players"],cid,obs["map"])

                t0_fake = time.perf_counter()-0.001  # always pass time check
                cands = _generate_candidates(
                    pos,my_r,my_c,my_radius,bombs_left,
                    obs["map"],obs["players"],cid,
                    danger_any,bomb_pos,danger_by_time,
                    extra_danger,combat_mode,nearby_armed,
                    obs["bombs"],t0_fake
                )

                # get rule agent's chosen action
                try: rule_a = agents[cid].act(obs)
                except: rule_a = STOP

                if len(cands) >= 2:
                    # encode context once, features per candidate
                    pos_hist = getattr(agents[cid], 'pos_history', [])  
                                    
                    is_stuck = (
                        len(pos_hist)>=16
                        and len(set(pos_hist))<=3
                        and len(obs["bombs"])==0
                        and pos not in danger_any
                    )
                    ctx = _encode_state_context(
                        my_r,my_c,my_radius,bombs_left,
                        obs["map"],obs["players"],cid,
                        danger_any,bomb_pos,danger_by_time,step
                    )
                    ctx[20] = float(is_stuck)
                    ctx[21] = float(combat_mode)

                    cand_feats = []
                    for a in cands:
                        cf = _encode_candidate_features(
                            a,my_r,my_c,my_radius,
                            obs["map"],obs["players"],cid,
                            danger_any,bomb_pos,danger_by_time,
                            obs["bombs"]
                        )
                        cand_feats.append((a,cf))

                    step_data[cid] = (ctx, cand_feats)

                actions[cid] = rule_a

            prev_obs = obs
            obs,terminated,truncated = env.step(actions)
            done = bool(terminated or truncated)
            step += 1

            # compute per-agent reward and push to buffer
            for cid in range(4):
                rew = _step_reward(prev_obs,obs,cid,actions[cid],terminated,truncated)
                if step_data[cid] is not None:
                    ctx,cand_feats = step_data[cid]
                    ep_bufs[cid].append((ctx,cand_feats,rew))

        # ── Compute discounted returns over RETURN_STEPS horizon ─────────
        for cid in range(4):
            buf = ep_bufs[cid]
            T   = len(buf)
            for t in range(T):
                # accumulate rewards t+1 … t+RETURN_STEPS
                ret = 0.0; g = 1.0
                for k in range(1, RETURN_STEPS+1):
                    if t+k >= T: break
                    _,_,r_k = buf[t+k]
                    ret += g*r_k; g*=GAMMA

                ctx, cand_feats, _ = buf[t]
                for a,cf in cand_feats:
                    all_states.append(ctx)
                    all_cands.append(cf)
                    all_returns.append(ret)

        if verbose and (gi+1) % 500 == 0:
            print(f"  collected game {gi+1}/{n_games} | "
                  f"samples so far: {len(all_returns):,}")

    states_arr  = np.array(all_states,  dtype=np.float32)
    cands_arr   = np.array(all_cands,   dtype=np.float32)
    returns_arr = np.array(all_returns, dtype=np.float32)

    print(f"\nDataset: {len(returns_arr):,} samples")
    print(f"  return range  [{returns_arr.min():.2f}, {returns_arr.max():.2f}]")
    print(f"  return mean   {returns_arr.mean():.3f}  std {returns_arr.std():.3f}")
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
    """Train ScorerNet. Returns trained model."""
    ret_norm,mu,sigma = normalise_returns(returns)
    trn_dl,val_dl    = build_dataloaders(states,cands,ret_norm)

    model     = ScorerNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=TRAIN_EPOCHS)
    loss_fn   = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nScorerNet: {n_params:,} parameters  device={DEVICE}")
    print(f"Train samples: {len(trn_dl.dataset):,}  "
          f"Val samples: {len(val_dl.dataset):,}\n")

    best_val = float("inf")
    for epoch in range(1,TRAIN_EPOCHS+1):
        # train
        model.train(); trn_loss=0.0; nb=0
        for X,Y in trn_dl:
            X,Y = X.to(DEVICE),Y.to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred,Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            trn_loss+=loss.item(); nb+=1
        trn_loss/=max(nb,1)

        # validate
        model.eval(); val_loss=0.0; nb=0
        with torch.no_grad():
            for X,Y in val_dl:
                X,Y = X.to(DEVICE),Y.to(DEVICE)
                val_loss += loss_fn(model(X),Y).item(); nb+=1
        val_loss/=max(nb,1)
        scheduler.step()

        if verbose:
            print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS} | "
                  f"trn={trn_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_PATH)
            if verbose: print(f"  ★ new best val={best_val:.4f} → {BEST_PATH}")

    # load best for final export
    model.load_state_dict(torch.load(BEST_PATH,map_location=DEVICE))
    model.eval()
    torch.save(model.state_dict(), SCORER_PATH)
    print(f"\nSaved trained scorer → {SCORER_PATH}")
    print(f"Best val MSE: {best_val:.4f}")
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
    Run n_games where agent 0 is hybrid, agents 1-3 are pure rule.
    Report win/draw/loss for the hybrid agent.
    """
    scorer.eval()
    wins=draws=losses=kills_t=boxes_t=0

    for gi in range(n_games):
        ms   = _EVAL_SEEDS[gi%N_EVAL_MAPS]
        cid  = 0   # hybrid is always agent 0
        env  = BomberEnv(max_steps=MAX_STEPS,seed=ms)
        obs  = env.reset()
        init_boxes = int(np.sum(obs["map"]==2))

        hybrid_ag = HybridAgent(cid, scorer)
        rule_ags  = {i:RuleAgent(i) for i in range(4) if i!=cid}

        done=False; step=0; kills=0
        while not done:
            if int(obs["players"][cid][2])!=1: break
            pe = sum(int(obs["players"][i][2]) for i in range(4) if i!=cid)
            acts=[0,0,0,0]
            acts[cid] = hybrid_ag.act(obs)
            for pid,ag in rule_ags.items(): acts[pid]=int(ag.act(obs))
            obs,terminated,truncated=env.step(acts)
            kills += max(0,pe-sum(int(obs["players"][i][2]) for i in range(4) if i!=cid))
            done=bool(terminated or truncated); step+=1

        alive = [int(p[2]) for p in obs["players"]]
        boxes_destroyed = init_boxes-int(np.sum(obs["map"]==2))
        if alive[cid]==1 and sum(alive)==1: wins+=1
        elif alive[cid]==1: draws+=1
        else: losses+=1
        kills_t+=kills; boxes_t+=boxes_destroyed

    ng=max(1,n_games)
    wr=wins/ng; dr=draws/ng
    print(f"\nEval ({n_games}g) | W={wins} D={draws} L={losses} | "
          f"WR={wr:.2%} | kills={kills_t/ng:.2f} boxes={boxes_t/ng:.0f}")
    return wr


# =============================================================================
# Iterative refinement  (optional)
# =============================================================================

def iterative_train(rounds=3, games_per_round=2000):
    """
    Multi-round refinement:
    Round 1: pure rule self-play → initial scorer
    Round 2+: hybrid self-play (hybrid vs rule mix) → refined scorer
    Each round loads previous scorer, collects new data, trains on combined set.
    """
    combined_states = np.zeros((0,STATE_DIM), dtype=np.float32)
    combined_cands  = np.zeros((0,CAND_DIM),  dtype=np.float32)
    combined_rets   = np.zeros((0,),           dtype=np.float32)

    best_wr = -1.0

    for rnd in range(1, rounds+1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{rounds}")
        print(f"{'='*60}")

        print(f"\n[{rnd}] Collecting {games_per_round} games...")
        s,c,r = collect_dataset(n_games=games_per_round)

        combined_states = np.concatenate([combined_states,s])
        combined_cands  = np.concatenate([combined_cands, c])
        combined_rets   = np.concatenate([combined_rets,  r])
        print(f"[{rnd}] Combined dataset: {len(combined_rets):,} samples")

        print(f"\n[{rnd}] Training...")
        scorer,_,_ = train(combined_states,combined_cands,combined_rets)

        print(f"\n[{rnd}] Evaluating...")
        wr = evaluate(scorer)

        if wr > best_wr:
            best_wr = wr
            torch.save(scorer.state_dict(), BEST_PATH)
            print(f"[{rnd}] ★ New best WR={wr:.2%} → {BEST_PATH}")

    # final export: best checkpoint
    best = ScorerNet().to(DEVICE)
    best.load_state_dict(torch.load(BEST_PATH,map_location=DEVICE))
    best.eval()
    torch.save(best.state_dict(), SCORER_PATH)
    print(f"\nFinal scorer exported → {SCORER_PATH}")
    return best


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ScorerNet for hybrid Bomberland agent")
    parser.add_argument("--n-games",       type=int,  default=N_GAMES,
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
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Config: {args.n_games} games | {args.epochs} epochs | "
          f"{args.rounds} rounds\n")

    # ── eval only ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(SCORER_PATH):
            print(f"ERROR: {SCORER_PATH} not found."); return
        scorer = ScorerNet().to(DEVICE)
        scorer.load_state_dict(torch.load(SCORER_PATH,map_location=DEVICE))
        scorer.eval()
        evaluate(scorer, n_games=args.n_eval)
        return

    # ── multi-round iterative ─────────────────────────────────────────────
    if args.rounds > 1:
        iterative_train(rounds=args.rounds,
                        games_per_round=args.n_games//args.rounds)
        return

    # ── single-pass ───────────────────────────────────────────────────────
    if args.skip_collect and os.path.exists(DATASET_PATH):
        print(f"Loading existing dataset from {DATASET_PATH}...")
        states,cands,returns = load_dataset()
    else:
        print(f"Collecting {args.n_games} self-play games...")
        states,cands,returns = collect_dataset(n_games=args.n_games)
        save_dataset(states,cands,returns)

    print(f"\nTraining for {args.epochs} epochs...")
    scorer,_,_ = train(states,cands,returns)

    print(f"\nEvaluating hybrid vs pure rule ({args.n_eval} games)...")
    evaluate(scorer, n_games=args.n_eval)

    print("\nDone. To use the scorer, place scorer.pt next to agent.py.")


if __name__ == "__main__":
    main()