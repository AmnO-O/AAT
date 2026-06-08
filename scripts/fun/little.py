"""
bomber_v10.py — Bomberland: Skill-Gated Curriculum with DAgger Teacher Forcing

═══════════════════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

The previous curriculum (v9) advances by round number regardless of whether
the agent actually mastered the current skill. This causes two failure modes:
  1. Agent advances to combat before it can reliably farm/escape → unlearns basics.
  2. Agent gets stuck if it happens to be slow at a skill — no remediation.

v10 replaces round-count gating with MASTERY GATING: the curriculum only
advances when the agent passes a measurable skill threshold. If it fails to
reach mastery within a maximum round budget, DAgger kicks in: we inject teacher
actions at a decaying mix rate, build a replay buffer of teacher demonstrations,
and mix that into the PPO update as an imitation loss — until the agent passes
the threshold and we reduce teacher mixing.

═══════════════════════════════════════════════════════════════════════════════
CURRICULUM PHASES
═══════════════════════════════════════════════════════════════════════════════

Phase 0  — BOX FARMING (solo)
  Teacher : BoxFarmerAgent
  Mastery : avg boxes destroyed ≥ BOX_MASTERY_THRESH AND survived ≥ SURVIVAL_THRESH
  Rounds  : up to MAX_ROUNDS_PER_PHASE before DAgger kicks in
  Signal  : pure box/item/survival reward, no kill signal

Phase 1  — BOMB ESCAPE (solo, own bombs)
  Teacher : TacticalRuleAgent (escape demonstrations)
  Mastery : survive ≥ ESCAPE_MASTERY_THRESH  AND  avg safe_bomb placement ≥ SAFEBOMB_THRESH
  Rounds  : up to MAX_ROUNDS_PER_PHASE
  Signal  : survival, safe-bomb placement, escape margin reward

Phase 2  — 1v1 COMBAT (vs SimpleRuleAgent, then SmarterRuleAgent)
  Teacher : SmarterRuleAgent / GeniusRuleAgent (attack demonstrations)
  Mastery : win-rate ≥ WINRATE_1V1_THRESH_SIMPLE, then ≥ WINRATE_1V1_THRESH_SMART
  Rounds  : up to MAX_ROUNDS_PER_PHASE each
  Signal  : full reward including kills

Phase 3  — 1v3 FULL GAME (3 active opponents, self-play mix)
  Teacher : None (DAgger not used — self-play drives learning)
  Mastery : win-rate ≥ WINRATE_FULL_THRESH or MAX_TOTAL_ROUNDS reached
  Rounds  : remaining budget; self-play league grows throughout

═══════════════════════════════════════════════════════════════════════════════
DAGGER MECHANICS
═══════════════════════════════════════════════════════════════════════════════

DAgger (Dataset Aggregation, Ross et al. 2011):
  - During rollout, with probability `dagger_beta` the agent follows the
    TEACHER's action instead of its own policy. (Beta decays as mastery rises.)
  - Every (state, teacher_action) pair is stored in a DAgger replay buffer.
  - In the PPO update, we add an imitation cross-entropy loss weighted by
    `dagger_coef` on samples drawn from the DAgger buffer.
  - As the agent masters each skill, beta and dagger_coef decay to 0.

This avoids compounding errors in early learning while preserving the RL signal.

═══════════════════════════════════════════════════════════════════════════════
KEY DIFFERENCES FROM v9
═══════════════════════════════════════════════════════════════════════════════

  ✅ Mastery gating  — curriculum advances on skill metrics, not round count
  ✅ DAgger          — teacher-forced rollout + imitation loss for struggling agents
  ✅ Phase 1 escape  — dedicated escape phase with survival rewards
  ✅ 1v1 mode        — 2 dummies, 1 opponent → clean kill signal
  ✅ Teacher mixing  — beta decays smoothly as mastery improves
  ✅ Mastery history — rolling window to avoid noisy single-eval decisions
  ✅ Phase metrics   — logged per phase for interpretability
  ✅ Checkpoint per phase — separate best_phase{N}.pth so no regressions

UNCHANGED FROM v9 (still correct):
  - 27-channel observation encoding
  - BomberNet architecture (spatial CNN + scalar MLP)
  - PPO hyperparameters (clip=0.20, gamma=0.98, lambda=0.95)
  - Fixed training/eval map seeds
  - League self-play pool
  - BC optional data anchoring
"""

import copy, os, random, sys, math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import json

sys.path.append(os.getcwd())
from engine.game import BomberEnv

# ─── Baseline imports ────────────────────────────────────────────────────────
def _try(mod, cls):
    try: return getattr(__import__(mod, fromlist=[cls]), cls)
    except: return None

TacticalRuleAgent = _try("agent.tactical_rule_agent", "TacticalRuleAgent")
GeniusRuleAgent   = _try("agent.genius_rule_agent",   "GeniusRuleAgent")
SmarterRuleAgent  = _try("agent.smarter_rule_agent",  "SmarterRuleAgent")
BoxFarmerAgent    = _try("agent.box_farmer_agent",    "BoxFarmerAgent")
SimpleRuleAgent   = _try("agent.simple_rule_agent",   "SimpleRuleAgent")
RandomAgent       = _try("agent.random_agent",        "RandomAgent")

# ─── Device / seed ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

# ─── Environment constants ────────────────────────────────────────────────────
BOARD_SIZE             = 13
INPUT_CHANNELS         = 27
NUM_ACTIONS            = 6
MAX_STEPS              = 500
EXPLOSION_TIME_HORIZON = 8.0

SPATIAL_CHANNELS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,21,24,25,26]  # 20
SCALAR_CHANNELS  = [14,17,18,19,20,22,23]                                  #  7

# ─── Fixed map pools ──────────────────────────────────────────────────────────
N_TRAIN_MAPS = 100
N_EVAL_MAPS  = 50
_TRAIN_SEEDS = [200_000 + i*137 for i in range(N_TRAIN_MAPS)]
_EVAL_SEEDS  = [800_000 + i*137 for i in range(N_EVAL_MAPS)]

# ─── PPO hyperparameters ──────────────────────────────────────────────────────
GAMES_PER_ROUND = 400
PPO_EPOCHS      = 4
PPO_BATCH       = 256
PPO_CLIP        = 0.20
GAMMA           = 0.98
LAM             = 0.95
VAL_COEF        = 0.5
ENT_INIT        = 0.01
ENT_DECAY       = 0.97
ENT_MIN         = 0.003
GRAD_CLIP       = 1.0
PPO_LR          = 3e-4
WD              = 1e-4
LEAGUE_SIZE     = 6
BC_MIX          = 0.05

# ─── Model paths ──────────────────────────────────────────────────────────────
MODEL_PATH   = "model_ppo.pth"
BEST_PATH    = "model_ppo_best.pth"
BC_TRAIN_DIR = "bc_train_chunks"
BC_MANIFEST  = "manifest.json"

# ─── DAgger hyperparameters ───────────────────────────────────────────────────
DAGGER_BETA_INIT  = 0.80   # probability of following teacher during rollout (decays)
DAGGER_BETA_MIN   = 0.05   # floor — always leave some RL exploration
DAGGER_COEF_INIT  = 0.30   # imitation loss weight in PPO update
DAGGER_COEF_MIN   = 0.02
DAGGER_DECAY      = 0.80   # multiply by this each round agent PASSES mastery check
DAGGER_BUFFER_MAX = 50_000 # max (state, action) pairs in DAgger buffer per phase

# ─── Mastery thresholds ───────────────────────────────────────────────────────
BOX_MASTERY_THRESH      = 28.0   # avg boxes destroyed per game (map ~40 boxes)
BOX_SURVIVAL_THRESH     = 0.90   # fraction of games survived in solo eval
ESCAPE_SURVIVAL_THRESH  = 0.85   # fraction survived after placing own bombs
SAFE_BOMB_THRESH        = 0.70   # fraction of PLACE_BOMB actions that are safe
WINRATE_1V1_SIMPLE      = 0.70   # 1v1 win-rate vs SimpleRuleAgent
WINRATE_1V1_SMARTER     = 0.55   # 1v1 win-rate vs SmarterRuleAgent
WINRATE_FULL            = 0.35   # win-rate in 4-player eval
MASTERY_WINDOW          = 3      # rolling evals that must ALL pass (avoids noise)
MAX_ROUNDS_PER_PHASE    = 25     # max rounds before DAgger activates if not mastered
MAX_TOTAL_ROUNDS        = 200    # hard cap across all phases

# ─── Phase IDs ────────────────────────────────────────────────────────────────
PHASE_BOX       = 0   # solo farming
PHASE_ESCAPE    = 1   # solo bomb-and-escape
PHASE_1V1_EASY  = 2   # 1v1 vs Simple
PHASE_1V1_HARD  = 3   # 1v1 vs Smarter
PHASE_FULL      = 4   # full 4-player game

PHASE_NAMES = {
    PHASE_BOX:      "BoxFarming",
    PHASE_ESCAPE:   "BombEscape",
    PHASE_1V1_EASY: "1v1-Simple",
    PHASE_1V1_HARD: "1v1-Smarter",
    PHASE_FULL:     "FullGame",
}

# ─── Seeding ──────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)
try: torch.set_float32_matmul_precision("high")
except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# BOARD HELPERS (identical to v9)
# ═══════════════════════════════════════════════════════════════════════════════
MOVES = {0:(0,0),1:(0,-1),2:(0,1),3:(-1,0),4:(1,0)}
def np_(p,a): dr,dc=MOVES[int(a)]; return p[0]+dr,p[1]+dc
def ib(r,c): return 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE
def pas(g,r,c): return ib(r,c) and int(g[r,c]) in (0,3,4)
def bset(bombs): return {(int(b[0]),int(b[1])) for b in bombs} if bombs is not None and len(bombs)>0 else set()
def brad(pl,o): return (1+int(pl[o][4])) if 0<=o<len(pl) and int(pl[o][2])==1 else 1

def blast_t(g,bx,by,r):
    t={(bx,by)}
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1,r+1):
            rx,cy=bx+dr*d,by+dc*d
            if not ib(rx,cy): break
            cv=int(g[rx,cy])
            if cv==1: break
            t.add((rx,cy))
            if cv==2: break
    return t

def blast_m(g,bx,by,r):
    m=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=bool); m[bx,by]=True
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1,r+1):
            rx,cy=bx+dr*d,by+dc*d
            if not ib(rx,cy): break
            cv=int(g[rx,cy])
            if cv==1: break
            m[rx,cy]=True
            if cv==2: break
    return m

def bet(g,pl,bombs):
    if bombs is None or len(bombs)==0: return np.zeros((0,),dtype=np.int32)
    n=len(bombs); t=np.array([max(0,int(b[2])) for b in bombs],dtype=np.int32)
    bl=[blast_t(g,int(bombs[i][0]),int(bombs[i][1]),brad(pl,int(bombs[i][3]) if bombs.shape[1]>3 else -1)) for i in range(n)]
    q=deque(range(n)); inq=[True]*n
    while q:
        i=q.popleft(); inq[i]=False; ti=int(t[i])
        for j in range(n):
            if i==j: continue
            if (int(bombs[j][0]),int(bombs[j][1])) in bl[i] and int(t[j])>ti:
                t[j]=ti
                if not inq[j]: q.append(j); inq[j]=True
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION PLANES (identical to v9)
# ═══════════════════════════════════════════════════════════════════════════════
def etp(g,pl,bombs,hz=EXPLOSION_TIME_HORIZON):
    p=np.ones((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    t=bet(g,pl,bombs); dn=hz if hz>0 else 1.0
    for i in range(len(bombs)):
        r=brad(pl,int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        nm=min(float(max(0,int(t[i]))),hz)/dn
        bm=blast_m(g,int(bombs[i][0]),int(bombs[i][1]),r)
        p[bm]=np.minimum(p[bm],nm)
    return p

def dng(g,pl,bombs,thr=1):
    if bombs is None or len(bombs)==0: return np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    p=etp(g,pl,bombs); return (p<=float(thr)/EXPLOSION_TIME_HORIZON).astype(np.float32)

def cdng(g,pl,bombs,ch=3):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    orig=np.array([max(0,int(b[2])) for b in bombs],dtype=np.int32); eff=bet(g,pl,bombs)
    for i in range(len(bombs)):
        e,o=int(eff[i]),int(orig[i])
        if e<=1 or e>ch or e>=o: continue
        r=brad(pl,int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        p[blast_m(g,int(bombs[i][0]),int(bombs[i][1]),r)]=1.0
    return p

def fdng(g,pl,bombs,hz=EXPLOSION_TIME_HORIZON):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    eff=bet(g,pl,bombs); dn=float(max(1.0,hz))
    for i in range(len(bombs)):
        r=brad(pl,int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        s=1.0-min(float(max(0,int(eff[i]))),dn)/dn
        if s<=0: continue
        bm=blast_m(g,int(bombs[i][0]),int(bombs[i][1]),r)
        p[bm]=np.maximum(p[bm],s)
    return p

def tet(g,pl,bombs):
    t=np.full((BOARD_SIZE,BOARD_SIZE),9999,dtype=np.int32)
    if bombs is None or len(bombs)==0: return t
    eff=bet(g,pl,bombs)
    for i,b in enumerate(bombs):
        r=brad(pl,int(b[3]) if bombs.shape[1]>3 else -1)
        bm=blast_m(g,int(b[0]),int(b[1]),r); t[bm]=np.minimum(t[bm],int(max(0,eff[i])))
    return t

def pp(g,pl,bombs,mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    for pid in range(4):
        if pid==mid or pid>=len(pl) or int(pl[pid][2])!=1 or int(pl[pid][3])<=0: continue
        r,c=int(pl[pid][0]),int(pl[pid][1])
        if ib(r,c): p[blast_m(g,r,c,1+int(pl[pid][4]))]=1.0
    return p

def fpp(g,pl,bombs,mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32); bk=bset(bombs)
    for pid in range(4):
        if pid==mid or pid>=len(pl) or int(pl[pid][2])!=1 or int(pl[pid][3])<=0: continue
        r,c=int(pl[pid][0]),int(pl[pid][1])
        if not ib(r,c): continue
        rad=1+int(pl[pid][4]); cands=[(r,c)]
        for a in (1,2,3,4):
            nr,nc=np_((r,c),a)
            if pas(g,nr,nc) and (nr,nc) not in bk: cands.append((nr,nc))
        for pr,pc in cands:
            bm=blast_m(g,pr,pc,rad); p[bm]=np.maximum(p[bm],0.5)
    return p

def btl(g,pl,bombs,mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if mid>=len(pl) or int(pl[mid][2])!=1: return p
    mr,mc=int(pl[mid][0]),int(pl[mid][1]); bk=bset(bombs)
    pv=np.isin(g,[0,3,4]).copy()
    for br,bc in bk:
        if ib(br,bc): pv[br,bc]=False
    def sh(a,dr,dc):
        o=np.zeros_like(a)
        if dr==-1: o[:-1,:]=a[1:,:]
        elif dr==1: o[1:,:]=a[:-1,:]
        elif dc==-1: o[:,:-1]=a[:,1:]
        elif dc==1: o[:,1:]=a[:,:-1]
        return o
    exits=sum(sh(pv.astype(np.int32),dr,dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
    xt=tet(g,pl,bombs); dn_=dng(g,pl,bombs,1); dangerous=(dn_>0)|(xt<=2)
    fragile=sum(sh((dangerous&pv).astype(np.int32),dr,dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
    p=np.where(exits==0,1.00,p); p=np.where((exits==1)&(fragile>=1),0.85,p)
    p=np.where((exits==1)&(fragile==0),0.65,p); p=np.where((exits==2)&(fragile>=2),0.40,p)
    p=np.where((exits==2)&(fragile<2),0.20,p); p=p*pv
    ri=np.arange(BOARD_SIZE)[:,None]; ci_=np.arange(BOARD_SIZE)[None,:]
    mh=np.abs(ri-mr)+np.abs(ci_-mc)
    p=np.maximum(p,np.where((mh<=1)&pv,0.75,0.0)); p=np.maximum(p,np.where((mh<=2)&pv,0.35,0.0))
    return p.astype(np.float32)

# ─── BFS helpers ─────────────────────────────────────────────────────────────
def esc_margin(g,pl,bombs,start,depth=6):
    xt=tet(g,pl,bombs); bk=bset(bombs); q=deque([(start,0)]); seen={start}; best=-9999
    while q:
        pos,d=q.popleft(); m=int(xt[pos[0],pos[1]])-d
        if m>best: best=m
        if d>=depth: continue
        for a in (1,2,3,4):
            n=np_(pos,a)
            if n in seen or n in bk or not pas(g,n[0],n[1]): continue
            seen.add(n); q.append((n,d+1))
    return -1.0 if best<-1000 else float(best)

def esc_score(g,pl,bombs,mid):
    if mid>=len(pl) or int(pl[mid][2])!=1: return 0.0
    pos=(int(pl[mid][0]),int(pl[mid][1])); m=esc_margin(g,pl,bombs,pos)
    return float(np.clip(m/6.0,0.0,1.0)) if m>0 else 0.0

def bfs_d(g,start,targets,bombs,depth=64):
    if not targets: return None
    bk=bset(bombs); q=deque([(start,0)]); seen={start}
    while q:
        pos,d=q.popleft()
        if pos in targets: return d
        if d>=depth: continue
        for a in (1,2,3,4):
            n=np_(pos,a)
            if n in seen or n in bk or not pas(g,n[0],n[1]): continue
            seen.add(n); q.append((n,d+1))
    return None

def bfs_r(g,start,bombs,depth=3):
    bk=bset(bombs); vis=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=bool)
    vis[start[0],start[1]]=True; q=deque([(start,0)]); cnt=0
    while q:
        pos,d=q.popleft()
        if d>0: cnt+=1
        if d>=depth: continue
        for a in (1,2,3,4):
            n=np_(pos,a)
            if not ib(n[0],n[1]) or vis[n[0],n[1]] or n in bk or not pas(g,n[0],n[1]): continue
            vis[n[0],n[1]]=True; q.append((n,d+1))
    return cnt

nd = lambda d,cap=24.0: 1.0 if d is None else float(min(d,cap))/cap
ns = lambda x,dn: float(np.clip(x/dn,0.0,1.0)) if dn>0 else 0.0

def legal_a(g,bombs,pos,bl):
    m=[0]; bk=bset(bombs)
    for a in (1,2,3,4):
        nr,nc=np_(pos,a)
        if pas(g,nr,nc) and (nr,nc) not in bk: m.append(a)
    if bl>0 and pos not in bk: m.append(5)
    return m

# ─── Bomb safety ─────────────────────────────────────────────────────────────
def add_hyp(bombs,pos,owner,timer=7):
    row=np.array([[pos[0],pos[1],timer,owner]],dtype=np.int8)
    return np.concatenate([bombs,row],0) if bombs is not None and len(bombs)>0 else row

def safe_b(g,pl,bombs,mid,pos,eib=False):
    if mid>=len(pl) or int(pl[mid][2])!=1 or not pas(g,pos[0],pos[1]): return False
    r=1+int(pl[mid][4]); hyp=add_hyp(bombs,pos,mid); bl=blast_t(g,pos[0],pos[1],r)
    bk=bset(hyp); thr=-1.0 if eib else 0.0
    for a in (1,2,3,4):
        nr,nc=np_(pos,a)
        if not pas(g,nr,nc) or (nr,nc) in bk or (nr,nc) in bl: continue
        if esc_margin(g,pl,hyp,(nr,nc))>thr: return True
    return False

def eib_c(g,pl,mid,pos,r):
    bl=blast_t(g,pos[0],pos[1],r)
    for i in range(4):
        if i==mid or i>=len(pl) or int(pl[i][2])!=1: continue
        if (int(pl[i][0]),int(pl[i][1])) in bl: return True
    return False

def sbp(g,pl,bombs,mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if mid>=len(pl) or int(pl[mid][2])!=1: return p
    r,c=int(pl[mid][0]),int(pl[mid][1])
    if not ib(r,c) or (r,c) in bset(bombs): return p
    rad=1+int(pl[mid][4]); bl=blast_t(g,r,c,rad)
    en={(int(pl[i][0]),int(pl[i][1])) for i in range(4) if i!=mid and i<len(pl) and int(pl[i][2])==1}
    if not any(int(g[x,y])==2 for x,y in bl) and not any(e in en for e in bl): return p
    hyp=add_hyp(bombs,(r,c),mid); bkh=bset(hyp); thr=-1.0 if any(e in en for e in bl) else 0.0
    for a in (1,2,3,4):
        nr,nc=np_((r,c),a)
        if not pas(g,nr,nc) or (nr,nc) in bkh or (nr,nc) in bl: continue
        if esc_margin(g,pl,hyp,(nr,nc))>thr: p[r,c]=1.0; break
    return p

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODE OBS — 27 channels (identical layout to v9)
# ═══════════════════════════════════════════════════════════════════════════════
def encode_obs(g,pl,bombs,mid,step):
    s=np.zeros((INPUT_CHANNELS,BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    s[0]=(g==1).astype(np.float32); s[1]=(g==2).astype(np.float32)
    s[2]=(g==0).astype(np.float32); s[3]=(g==3).astype(np.float32); s[4]=(g==4).astype(np.float32)
    for pid in range(4):
        if pid<len(pl) and int(pl[pid][2])==1:
            r,c=int(pl[pid][0]),int(pl[pid][1])
            if ib(r,c): s[5+pid,r,c]=1.0
    s[9]=etp(g,pl,bombs); s[10]=dng(g,pl,bombs,1); s[11]=cdng(g,pl,bombs); s[12]=fdng(g,pl,bombs)
    me=0; mpos=(0,0); bl_=0
    if mid<len(pl) and int(pl[mid][2])==1:
        me=1; mr,mc=int(pl[mid][0]),int(pl[mid][1]); mpos=(mr,mc)
        if ib(mr,mc): s[13,mr,mc]=1.0
        bl_=int(pl[mid][3])
    s[14].fill(ns(bl_,5.0))
    if bombs is not None and len(bombs)>0:
        eff=bet(g,pl,bombs)
        for i in range(len(bombs)):
            r,c=int(bombs[i][0]),int(bombs[i][1])
            if not ib(r,c): continue
            t=max(int(eff[i]),1); s[15,r,c]=max(s[15,r,c],1.0/t)
            ow=int(bombs[i][3]) if bombs.shape[1]>3 else -1
            s[16,r,c]=max(s[16,r,c],ns(brad(pl,ow),6.0))
    if me:
        ip={(int(r),int(c)) for r,c in np.argwhere((g==3)|(g==4))}
        ep={(int(pl[i][0]),int(pl[i][1])) for i in range(4) if i!=mid and i<len(pl) and int(pl[i][2])==1}
        s[17].fill(nd(bfs_d(g,mpos,ip,bombs))); s[18].fill(nd(bfs_d(g,mpos,ep,bombs)))
        s[19].fill(ns(bfs_r(g,mpos,bombs,3),20.0)); s[20].fill(esc_score(g,pl,bombs,mid))
        s[21]=sbp(g,pl,bombs,mid)
    else: s[17].fill(1.0); s[18].fill(1.0)
    s[22].fill(ns(len(bombs) if bombs is not None else 0,10.0))
    s[23].fill(ns(step,float(MAX_STEPS)))
    s[24]=pp(g,pl,bombs,mid); s[25]=fpp(g,pl,bombs,mid); s[26]=btl(g,pl,bombs,mid)
    return torch.from_numpy(s)

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK — BomberNet (identical to v9, copy to agent.py for inference)
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout)
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        return torch.relu(self.bn2(self.conv2(h)) + x)

_HEAD_CONV_CH = 8  # 8×7×7 + 7 scalars = 399 feat_dim

class BomberNet(nn.Module):
    """
    375K params. Copy BomberNet + ResidualBlock + _HEAD_CONV_CH into agent.py.
    """
    _SPATIAL = SPATIAL_CHANNELS; _SCALAR = SCALAR_CHANNELS; _POOL = 7
    def __init__(self, input_channels=INPUT_CHANNELS, num_actions=NUM_ACTIONS, width=64):
        super().__init__()
        n_sp, n_sc = len(self._SPATIAL), len(self._SCALAR)
        pool_sz  = self._POOL
        feat_dim = _HEAD_CONV_CH * pool_sz * pool_sz + n_sc  # 8×49+7 = 399
        self.stem = nn.Sequential(
            nn.Conv2d(n_sp, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, 0.05), ResidualBlock(width, 0.05), ResidualBlock(width, 0.05))
        self.pool        = nn.AdaptiveAvgPool2d(pool_sz)
        self.policy_conv = nn.Conv2d(width, _HEAD_CONV_CH, 1)
        self.value_conv  = nn.Conv2d(width, _HEAD_CONV_CH, 1)
        self.policy_head = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim,128), nn.ReLU(True), nn.Dropout(0.05), nn.Linear(128, num_actions))
        self.value_head  = nn.Sequential(nn.Flatten(), nn.Linear(feat_dim,128), nn.ReLU(True), nn.Dropout(0.02), nn.Linear(128, 1))
        self.register_buffer("_sp_idx", torch.tensor(self._SPATIAL, dtype=torch.long))
        self.register_buffer("_sc_idx", torch.tensor(self._SCALAR,  dtype=torch.long))
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01); nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.orthogonal_(self.value_head[-1].weight,  gain=1.0);  nn.init.zeros_(self.value_head[-1].bias)
    def forward(self, x):
        sp = x[:, self._sp_idx]; sc = x[:, self._sc_idx, 0, 0]
        feat = self.pool(self.blocks(self.stem(sp)))
        p = torch.relu(self.policy_conv(feat))
        v = torch.relu(self.value_conv(feat))
        logits = self.policy_head(torch.cat([p.flatten(1), sc], 1))
        value  = self.value_head(torch.cat([v.flatten(1), sc], 1)).squeeze(-1)
        return logits, value

def fwd(m,x): return m(x)

# ═══════════════════════════════════════════════════════════════════════════════
# ACTION MASKING
# ═══════════════════════════════════════════════════════════════════════════════
def lmask(g,bombs,pos,bl):
    m=np.zeros(NUM_ACTIONS,dtype=np.float32)
    for a in legal_a(g,bombs,pos,bl): m[a]=1.0
    if m.sum()<=0: m[0]=1.0
    return m

def smask(g,pl,bombs,mid,lm):
    """Shield mask — blocks clearly suicidal actions. Used from Phase 1 onward."""
    m=lm.copy()
    if mid>=len(pl) or int(pl[mid][2])!=1:
        if m.sum()<=0: m[0]=1.0; return m
    pos=(int(pl[mid][0]),int(pl[mid][1])); bk=bset(bombs)
    d1=dng(g,pl,bombs,1); d2=dng(g,pl,bombs,2)
    in_d=bool(d1[pos[0],pos[1]]>0 or d2[pos[0],pos[1]]>0)
    if in_d:
        safe=[]
        for a in (1,2,3,4):
            if m[a]<=0: continue
            nr,nc=np_(pos,a)
            if not pas(g,nr,nc) or (nr,nc) in bk: m[a]=0.0; continue
            if esc_margin(g,pl,bombs,(nr,nc))>0: safe.append(a)
            else: m[a]=0.0
        if safe: m[0]=0.0
        elif m[0]<=0: m[0]=1.0
    else:
        if m[5]>0:
            r=1+int(pl[mid][4]); eib=eib_c(g,pl,mid,pos,r)
            if not safe_b(g,pl,bombs,mid,pos,eib): m[5]=0.0
    if m.sum()<=0: m[0]=1.0
    return m

def sample_a(model,st,mask,stoch=True,temp=1.0):
    logits,val=fwd(model,st); logits=logits/max(float(temp),1e-6)
    mt=torch.tensor(mask,dtype=torch.bool,device=logits.device).unsqueeze(0)
    ml=logits.clone(); ml[~mt]=-1e9
    dist=Categorical(logits=ml); a=dist.sample() if stoch else torch.argmax(ml,-1)
    return int(a.item()),float(dist.log_prob(a).item()),float(dist.entropy().item()),float(val.item())
def _teacher_fallback_action(obs, aid, phase, raw_action):
    """Convert a baseline teacher action into a safe, legal demonstration action."""
    try:
        players = obs["players"]
        g = obs["map"]
        bombs = obs["bombs"]
        if aid >= len(players) or int(players[aid][2]) != 1:
            return 0
        pos = (int(players[aid][0]), int(players[aid][1]))
        bl_ = int(players[aid][3])
        lm = lmask(g, bombs, pos, bl_)
        sm = smask(g, players, bombs, aid, lm)
        if 0 <= int(raw_action) < NUM_ACTIONS and sm[int(raw_action)] > 0:
            return int(raw_action)

        candidates = [a for a in range(NUM_ACTIONS) if sm[a] > 0]
        if not candidates:
            return 0

        # Prefer a safe bomb if it is actually useful.
        if 5 in candidates and safe_b(g, players, bombs, aid, pos, False):
            return 5

        danger = bool(dng(g, players, bombs, 1)[pos[0], pos[1]] > 0 or dng(g, players, bombs, 2)[pos[0], pos[1]] > 0)
        box_targets = {(int(r), int(c)) for r, c in np.argwhere((g == 2) | (g == 3) | (g == 4))}
        enemy_targets = {(int(players[i][0]), int(players[i][1])) for i in range(4)
                         if i != aid and i < len(players) and int(players[i][2]) == 1}

        def score(a):
            if a == 0:
                return -0.2 if danger else 0.0
            nr, nc = np_(pos, a)
            if not pas(g, nr, nc):
                return -999.0
            s = 0.0
            if danger:
                s += esc_margin(g, players, bombs, (nr, nc))
            if phase == PHASE_BOX:
                d = bfs_d(g, (nr, nc), box_targets, bombs, depth=20)
                s += 0.6 if d is None else max(0.0, 6.0 - float(d))
            elif phase == PHASE_ESCAPE:
                s += 1.0 * max(0.0, esc_margin(g, players, bombs, (nr, nc)))
            elif phase in (PHASE_1V1_EASY, PHASE_1V1_HARD, PHASE_FULL):
                d = bfs_d(g, (nr, nc), enemy_targets, bombs, depth=20)
                s += 0.4 if d is None else max(0.0, 8.0 - float(d))
            return s

        return int(max(candidates, key=score))
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════
class StopAgent:
    """Permanently does nothing."""
    def __init__(self,aid): self.aid=int(aid)
    def act(self,obs): return 0

class FrozenAgent:
    """Frozen snapshot of our own policy — used for self-play opponents."""
    def __init__(self,aid,model,det=True):
        self.aid=int(aid); self.model=model; self.det=bool(det); self._s=0
    def reset(self): self._s=0
    def act(self,obs):
        if self.aid>=len(obs["players"]) or int(obs["players"][self.aid][2])!=1:
            self._s+=1; return 0
        s=self._s; self._s+=1
        dev=next(self.model.parameters()).device
        st=encode_obs(obs["map"],obs["players"],obs["bombs"],self.aid,s).unsqueeze(0).to(dev)
        pos=(int(obs["players"][self.aid][0]),int(obs["players"][self.aid][1]))
        bl_=int(obs["players"][self.aid][3])
        lm_=lmask(obs["map"],obs["bombs"],pos,bl_)
        sm_=smask(obs["map"],obs["players"],obs["bombs"],self.aid,lm_)
        with torch.no_grad(): a,_,_,_=sample_a(self.model,st,sm_,stoch=not self.det)
        return a

class TeacherAgent:
    """
    Wraps a baseline rule agent as a teacher.
    The raw teacher output is post-processed through the same legality / safety
    checks we use for inference, so DAgger never records obviously broken actions.
    """
    def __init__(self, aid, cls, phase):
        self.aid = int(aid)
        self.phase = int(phase)
        self._inner = cls(aid) if cls is not None else StopAgent(aid)

    def act(self, obs):
        try:
            raw = int(self._inner.act(obs))
        except Exception:
            raw = 0
        return _teacher_fallback_action(obs, self.aid, self.phase, raw)

# ═══════════════════════════════════════════════════════════════════════════════
# DAGGER BUFFER
# ═══════════════════════════════════════════════════════════════════════════════
class DAggerBuffer:
    """
    Circular buffer of (state_tensor, teacher_action) pairs for imitation.
    States stored as float32 numpy arrays (INPUT_CHANNELS, 13, 13).
    """
    def __init__(self, maxlen=DAGGER_BUFFER_MAX):
        self.maxlen = maxlen
        self.states:  List[np.ndarray] = []
        self.actions: List[int]        = []
        self._idx = 0

    def add(self, state_np: np.ndarray, action: int):
        if len(self.states) < self.maxlen:
            self.states.append(state_np)
            self.actions.append(action)
        else:
            self.states[self._idx]  = state_np
            self.actions[self._idx] = action
            self._idx = (self._idx + 1) % self.maxlen

    def sample(self, n: int):
        """Return (states, actions) tensors of size min(n, len)."""
        k = min(n, len(self.states))
        if k == 0: return None, None
        idxs = np.random.choice(len(self.states), k, replace=False)
        s = torch.tensor(np.stack([self.states[i] for i in idxs]), dtype=torch.float32)
        a = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        return s, a

    def __len__(self): return len(self.states)

    def clear(self): self.states.clear(); self.actions.clear(); self._idx=0

# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUE
# ═══════════════════════════════════════════════════════════════════════════════
class League:
    def __init__(self,n=LEAGUE_SIZE): self.n=n; self.snaps=[]
    def add(self,m): s=copy.deepcopy(m).cpu().eval(); self.snaps.append(s); self.snaps=self.snaps[-self.n:]
    def sample(self): return random.choice(self.snaps) if self.snaps else None

# ═══════════════════════════════════════════════════════════════════════════════
# OPPONENT POOLS
# ═══════════════════════════════════════════════════════════════════════════════
def _pool(specs):
    p=[]
    for cls,w in specs:
        if cls: p.extend([cls]*w)
    return p or [StopAgent]

_RAND = _pool([(RandomAgent,1)])
_WEAK = _pool([(SimpleRuleAgent,2),(BoxFarmerAgent,2)])
_MED  = _pool([(SmarterRuleAgent,2),(BoxFarmerAgent,1),(SimpleRuleAgent,1)])
_STR  = _pool([(TacticalRuleAgent,4),(GeniusRuleAgent,3),(SmarterRuleAgent,2)])
_EVAL = _pool([(TacticalRuleAgent,2),(GeniusRuleAgent,2),(SmarterRuleAgent,2),(BoxFarmerAgent,1),(SimpleRuleAgent,1)])

def make_eval_opps(cid, seed):
    rng=random.Random(seed)
    return {pid: rng.choice(_EVAL)(pid) for pid in range(4) if pid != cid}

# ═══════════════════════════════════════════════════════════════════════════════
# OPPONENT BUILDER — phase-aware
# ═══════════════════════════════════════════════════════════════════════════════
def make_opps_for_phase(cid, opp_seed, frozen, league, phase):
    """
    Build the opponent dict based on current curriculum phase.

    Phase 0 (BOX):      3 x StopAgent — clean farming signal
    Phase 1 (ESCAPE):   3 x StopAgent — no threats, just own bombs
    Phase 2 (1v1 Easy): 1 x SimpleRuleAgent, 2 x StopAgent
    Phase 3 (1v1 Hard): 1 x SmarterRuleAgent (70%) / GeniusRuleAgent (30%), 2 x Stop
    Phase 4 (Full):     3 active: mix strong baselines + self-play (50% frozen/league)
    """
    rng = random.Random(opp_seed)
    other_ids = [p for p in range(4) if p != cid]
    opps = {}

    if phase in (PHASE_BOX, PHASE_ESCAPE):
        for pid in other_ids:
            opps[pid] = StopAgent(pid)
        return opps

    if phase == PHASE_1V1_EASY:
        active_pid = rng.choice(other_ids)
        for pid in other_ids:
            opps[pid] = SimpleRuleAgent(pid) if pid == active_pid else StopAgent(pid)
        return opps

    if phase == PHASE_1V1_HARD:
        active_pid = rng.choice(other_ids)
        for pid in other_ids:
            if pid == active_pid:
                cls = SmarterRuleAgent if (SmarterRuleAgent and rng.random()<0.7) else GeniusRuleAgent
                opps[pid] = (cls or SmarterRuleAgent or SimpleRuleAgent)(pid)
            else:
                opps[pid] = StopAgent(pid)
        return opps

    # Phase FULL: all 3 active — mix strong baselines + self-play
    p_frz, p_lgu = 0.35, 0.15
    for pid in other_ids:
        r = rng.random()
        if r < p_frz and frozen is not None:
            fa = FrozenAgent(pid, frozen, det=rng.random()<0.6); fa.reset(); opps[pid] = fa
        elif r < p_frz+p_lgu and league is not None and league.snaps:
            lm = league.sample().to(DEVICE)
            fa = FrozenAgent(pid, lm, det=rng.random()<0.5); fa.reset(); opps[pid] = fa
        else:
            opps[pid] = rng.choice(_STR)(pid)
    return opps

def make_teacher_for_phase(cid, phase):
    """
    Return a TeacherAgent appropriate for the current phase.
    Returns None if no teacher is active for this phase.
    """
    if phase == PHASE_BOX:
        cls = BoxFarmerAgent
    elif phase == PHASE_ESCAPE:
        cls = TacticalRuleAgent  # great at safe bomb + escape
    elif phase == PHASE_1V1_EASY:
        cls = SmarterRuleAgent   # shows attack patterns
    elif phase == PHASE_1V1_HARD:
        cls = GeniusRuleAgent
    else:
        return None  # no teacher in full game phase
    if cls is None:
        return None
    return TeacherAgent(cid, cls, phase)

# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION — phase-aware
# ═══════════════════════════════════════════════════════════════════════════════
def reward_fn(prev_obs, next_obs, my_id, action, terminated, truncated, phase):
    """
    Phase-aware reward:
      Phase 0 (BOX):    heavy box/item bonus, survival, no kill signal
      Phase 1 (ESCAPE): survival is primary; reward safe bomb placement & escape margin
      Phase 2/3 (1v1):  kill signal prominent, box moderate, death penalized
      Phase 4 (FULL):   full reward — all signals active
    """
    r = 0.0
    pp_ = prev_obs["players"]; np__ = next_obs["players"]
    pm  = prev_obs["map"];     nm   = next_obs["map"]

    # ── Survival / death ──────────────────────────────────────────────────────
    if my_id < len(pp_) and my_id < len(np__):
        pa, na = int(pp_[my_id][2]), int(np__[my_id][2])
        if pa == 1 and na == 1:
            r += 0.0002   # small tick bonus for staying alive
        elif pa == 1 and na == 0:
            # Phase 0/1: dying is catastrophic — aggressive penalty
            death_pen = -6.0 if phase in (PHASE_BOX, PHASE_ESCAPE) else -4.0
            r += death_pen
        # Item collection
        if pa == 1 and na == 1:
            if int(np__[my_id][4]) > int(pp_[my_id][4]):
                r += 0.08   # radius item
            npos = (int(np__[my_id][0]), int(np__[my_id][1]))
            if ib(npos[0], npos[1]):
                pc = int(pm[npos[0], npos[1]]); nc_ = int(nm[npos[0], npos[1]])
                if pc in (3,4) and nc_ == 0:
                    r += 0.08 if pc == 3 else 0.10  # capacity item slightly better

    # ── Kills (only meaningful Phase 2+) ──────────────────────────────────────
    if phase >= PHASE_1V1_EASY:
        pe = int(np.sum(pp_[:,2])) - int(pp_[my_id][2]) if my_id < len(pp_) else 0
        ne = int(np.sum(np__[:,2])) - int(np__[my_id][2]) if my_id < len(np__) else 0
        kills = max(0, pe - ne)
        if kills > 0:
            last = (ne == 0)
            r += (3.5 if last else 2.0) * kills

    # ── Box destruction ────────────────────────────────────────────────────────
    boxes = max(0, int(np.sum(pm==2)) - int(np.sum(nm==2)))
    if boxes > 0:
        if phase == PHASE_BOX:
            # Strong farming signal — primary learning objective
            r += 0.12 * boxes + (0.02*(boxes-1) if boxes >= 2 else 0)
        elif phase == PHASE_ESCAPE:
            # Moderate — farming is secondary to survival in this phase
            r += 0.05 * boxes
        else:
            # Combat phases: boxes matter but kills dominate
            r += 0.03 * boxes + (0.01*(boxes-1) if boxes >= 2 else 0)

    # ── Bomb placement quality ────────────────────────────────────────────────
    if action == 5 and my_id < len(pp_) and int(pp_[my_id][2]) == 1:
        pos = (int(pp_[my_id][0]), int(pp_[my_id][1]))
        rad = 1 + int(pp_[my_id][4])
        bl  = blast_t(pm, pos[0], pos[1], rad)
        hit_e = sum(1 for i in range(4) if i != my_id and i < len(pp_)
                    and int(pp_[i][2]) == 1 and (int(pp_[i][0]), int(pp_[i][1])) in bl)
        eib = hit_e > 0

        if safe_b(pm, pp_, prev_obs["bombs"], my_id, pos, eib):
            # Good bomb: can escape after placing
            r += 0.08
            if phase >= PHASE_1V1_EASY:
                r += 0.30 * hit_e                               # targeting reward
            r += 0.015 * sum(1 for rx,cx in bl if int(pm[rx,cx]) == 2)  # box hits
            # Chain reaction bonus
            hyp = add_hyp(prev_obs["bombs"], pos, my_id)
            bef = bet(pm, pp_, prev_obs["bombs"]); aft = bet(pm, pp_, hyp)
            if len(bef) and len(aft):
                r += 0.004 * float(np.sum(np.maximum(0, bef - aft)))
        else:
            # Unsafe bomb: stepped in own blast, no escape
            # Harsher penalty in escape phase so agent learns NOT to do this
            r -= 0.20 if phase == PHASE_ESCAPE else 0.12

    # ── Phase 1: reward escape margin while in danger ──────────────────────────
    if phase == PHASE_ESCAPE and my_id < len(np__) and int(np__[my_id][2]) == 1:
        pos = (int(np__[my_id][0]), int(np__[my_id][1]))
        margin = esc_margin(nm, np__, next_obs["bombs"], pos)
        if margin > 0:
            r += 0.005 * min(margin, 6.0)    # small continuous reward for being safe

    # ── Anti-stall ────────────────────────────────────────────────────────────
    r -= 0.001

    # ── Terminal bonus/penalty ────────────────────────────────────────────────
    if terminated or truncated:
        if my_id < len(np__) and int(np__[my_id][2]) == 1:
            alive_count = int(np.sum(np__[:,2]))
            if alive_count == 1:
                r += 10.0   # sole survivor: win
            elif phase in (PHASE_BOX, PHASE_ESCAPE):
                # In solo phases: surviving to 500 is the target
                r += 5.0
            else:
                r += 0.5    # survived but not last — draw bonus
        else:
            r -= 1.5

    return float(np.clip(r, -12.0, 15.0))

# ═══════════════════════════════════════════════════════════════════════════════
# ROLLOUT COLLECTION — with DAgger teacher injection
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Ep:
    states:  List[np.ndarray] = field(default_factory=list)
    actions: List[int]        = field(default_factory=list)
    rewards: List[float]      = field(default_factory=list)
    dones:   List[bool]       = field(default_factory=list)
    lps:     List[float]      = field(default_factory=list)
    vals:    List[float]      = field(default_factory=list)
    masks:   List[np.ndarray] = field(default_factory=list)
    last_val: float           = 0.0

def _gae(ep):
    T=len(ep.rewards); adv=np.zeros(T,dtype=np.float32); v=np.array(ep.vals,dtype=np.float32); g=0.0
    for t in reversed(range(T)):
        nv=0.0 if ep.dones[t] else (v[t+1] if t+1<T else ep.last_val)
        delta=ep.rewards[t]+GAMMA*nv-v[t]; g=delta+GAMMA*LAM*(1.0-float(ep.dones[t]))*g; adv[t]=g
    return adv,adv+v

def flatten_eps(eps):
    S,A,LP,R,ADV,M=[],[],[],[],[],[]
    for ep in eps:
        if not ep.states: continue
        adv,ret=_gae(ep)
        S.extend(ep.states); A.extend(ep.actions); LP.extend(ep.lps)
        R.extend(ret.tolist()); ADV.extend(adv.tolist()); M.extend(ep.masks)
    if not S: raise RuntimeError("Empty rollout.")
    mk=lambda l,dt: torch.tensor(np.array(l),dtype=dt)
    st=mk(S,torch.float32); at=mk(A,torch.long); lpt=mk(LP,torch.float32)
    rt=mk(R,torch.float32); mt=mk(M,torch.float32)
    advt=mk(ADV,torch.float32)
    advt=(advt-advt.mean())/(advt.std()+1e-8)
    return st,at,lpt,rt,advt,mt

def collect(
    model, frozen, n_games, phase, league,
    dagger_buffer: Optional[DAggerBuffer] = None,
    dagger_beta: float = 0.0,
    safe_bomb_counter: Optional[List] = None,  # [safe_count, total_bomb_count]
):
    """
    Rollout collection with DAgger teacher injection.

    dagger_beta : probability of using teacher action instead of policy action.
                  0.0 = pure RL. 1.0 = pure teacher.
    dagger_buffer: if not None, (state, teacher_action) pairs are appended here.
    safe_bomb_counter: mutable list [safe, total] to track bomb safety stats.
    """
    model.eval()
    if frozen is not None: frozen.eval()

    # Phase 0 and 1: no shield mask during training (learn safety from reward)
    use_shield = phase not in (PHASE_BOX, PHASE_ESCAPE)
    eps = []
    if safe_bomb_counter is None: safe_bomb_counter = [0, 0]

    for gi in range(n_games):
        map_seed = _TRAIN_SEEDS[gi % N_TRAIN_MAPS]
        opp_seed = (map_seed + phase * 999_983 + gi * 1_000_003) & 0x7FFFFFFF
        cid = gi % 4

        env = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs = env.reset()
        opps = make_opps_for_phase(cid, opp_seed, frozen, league, phase)

        # DAgger teacher for this agent
        teacher = make_teacher_for_phase(cid, phase) if dagger_beta > 0.0 else None

        ep = Ep(); done = False; step = 0; ta = False

        while not done:
            if cid >= len(obs["players"]) or int(obs["players"][cid][2]) != 1: break

            st = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            st_np = st.squeeze(0).cpu().numpy().astype(np.float32)
            pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl_ = int(obs["players"][cid][3])

            lm = lmask(obs["map"], obs["bombs"], pos, bl_)
            train_mask = smask(obs["map"], obs["players"], obs["bombs"], cid, lm) if use_shield else lm

            # ── DAgger: get teacher action for logging (and possible override) ──
            teacher_a = None
            if teacher is not None:
                try: teacher_a = int(teacher.act(obs))
                except: teacher_a = 0

            # ── DAgger buffer: store (state, teacher_action) regardless of who acts ─
            if teacher_a is not None and dagger_buffer is not None:
                dagger_buffer.add(st_np, teacher_a)

            # ── Choose action: teacher or policy ──────────────────────────────
            use_teacher_action = (teacher_a is not None) and (random.random() < dagger_beta)
            with torch.no_grad():
                a, lp, _, val = sample_a(model, st, train_mask, stoch=True, temp=1.0)

            if use_teacher_action:
                # Override with teacher's action; re-compute log prob under current policy
                a = teacher_a
                with torch.no_grad():
                    logits, val_t = fwd(model, st)
                    mt = torch.tensor(train_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
                    ml = logits.clone(); ml[~mt] = -1e9
                    dist = Categorical(logits=ml)
                    a_t = torch.tensor([a], device=logits.device)
                    # Clamp log prob to avoid -inf if teacher chose masked action
                    lp = float(dist.log_prob(a_t).clamp(-10.0, 0.0).item())
                    val = float(val_t.item())

            # ── Track safe bomb stats ─────────────────────────────────────────
            if a == 5 and my_id_alive(obs, cid):
                safe_bomb_counter[1] += 1
                if safe_b(obs["map"], obs["players"], obs["bombs"], cid, pos, False):
                    safe_bomb_counter[0] += 1

            # ── Step env ──────────────────────────────────────────────────────
            acts = [0,0,0,0]; acts[cid] = a
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            prev_obs = obs
            obs, terminated, truncated = env.step(acts)
            died = int(obs["players"][cid][2]) == 0
            rew  = reward_fn(prev_obs, obs, cid, a, terminated, truncated, phase)
            gd   = bool(died or terminated)

            ep.states.append(st_np)
            ep.actions.append(a)
            ep.rewards.append(float(rew))
            ep.dones.append(gd)
            ep.lps.append(lp)
            ep.vals.append(float(val))
            ep.masks.append(train_mask.astype(np.float32))

            ta   = bool(truncated and not terminated and not died)
            done = bool(terminated or truncated or died)
            step += 1

        # Bootstrap value for truncated episodes
        if ta and ep.states:
            try:
                ls = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): _, lv = fwd(model, ls); ep.last_val = float(lv.item())
            except: pass

        if ep.states: eps.append(ep)
        if (gi+1) % 50 == 0:
            print(f"  Rollout {gi+1}/{n_games} | eps={len(eps)} | steps={sum(len(e.states) for e in eps)}", flush=True)

    return eps

def my_id_alive(obs, cid):
    return cid < len(obs["players"]) and int(obs["players"][cid][2]) == 1

# ═══════════════════════════════════════════════════════════════════════════════
# BC DATASET (optional anchor — same as v9)
# ═══════════════════════════════════════════════════════════════════════════════
class BCDataset(IterableDataset):
    def __init__(self,d):
        super().__init__(); self.d=d
        try:
            p=os.path.join(d,"manifest.json"); m=json.load(open(p))
            self.chunks=list(m.get("chunks",[])); self.n=sum(c.get("count",0) for c in self.chunks)
        except: self.chunks=[]; self.n=0
    def __len__(self): return self.n
    def __iter__(self):
        info=get_worker_info(); wid=0 if info is None else info.id; nw=1 if info is None else info.num_workers
        rng=np.random.default_rng(SEED+wid*1337); idxs=np.arange(len(self.chunks))
        rng.shuffle(idxs)
        for ci in idxs[wid::nw]:
            try:
                data=np.load(os.path.join(self.d,self.chunks[int(ci)]["file"]))
                st=data["states"]; ac=data["actions"]; order=np.arange(len(ac)); rng.shuffle(order)
                for i in order:
                    s=torch.from_numpy(st[int(i)]).float(); a=int(ac[int(i)])
                    yield s,torch.tensor(a,dtype=torch.long)
            except: continue

# ═══════════════════════════════════════════════════════════════════════════════
# PPO UPDATE — with DAgger imitation loss
# ═══════════════════════════════════════════════════════════════════════════════
def ppo_update(
    model, eps, optimizer,
    bc_loader=None, ent_coef=ENT_INIT,
    dagger_buffer: Optional[DAggerBuffer] = None,
    dagger_coef: float = 0.0,
):
    if not eps: return
    states, actions, old_lps, returns, advantages, masks = flatten_eps(eps)
    N = states.shape[0]; model.train()
    bc_iter = iter(bc_loader) if bc_loader else None

    for epoch in range(1, PPO_EPOCHS+1):
        idxs = np.random.permutation(N)
        tp=tv=te=tt=tdg=nb=0.0
        for s in range(0, N, PPO_BATCH):
            bi = idxs[s:s+PPO_BATCH]
            if len(bi) == 0: continue
            bs   = states[bi].to(DEVICE)
            ba   = actions[bi].to(DEVICE)
            blp  = old_lps[bi].to(DEVICE)
            brt  = returns[bi].to(DEVICE)
            bad  = advantages[bi].to(DEVICE)
            bm   = masks[bi].to(DEVICE)

            logits, values = fwd(model, bs)
            ml = logits.clone(); ml[bm<=0] = -1e9
            dist = Categorical(logits=ml)
            nlp  = dist.log_prob(ba)
            ent  = dist.entropy().mean()
            ratio = torch.exp(nlp - blp)
            clp   = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP)
            pl    = -torch.mean(torch.min(ratio*bad, clp*bad))
            vl    = torch.mean((values - brt)**2)
            loss  = pl + VAL_COEF*vl - ent_coef*ent

            # ── BC anchor loss (optional) ──────────────────────────────────────
            if bc_iter is not None and BC_MIX > 0:
                try: bcs, bca = next(bc_iter)
                except StopIteration:
                    bc_iter = iter(bc_loader) if bc_loader else None
                    bcs = None
                    if bc_iter:
                        try: bcs, bca = next(bc_iter)
                        except: pass
                if bcs is not None:
                    bcs = bcs.to(DEVICE); bca = bca.to(DEVICE)
                    bcl, _ = fwd(model, bcs)
                    loss += BC_MIX * nn.functional.cross_entropy(bcl, bca)

            # ── DAgger imitation loss ─────────────────────────────────────────
            dagger_loss_val = 0.0
            if dagger_buffer is not None and dagger_coef > 0 and len(dagger_buffer) > 0:
                ds, da = dagger_buffer.sample(PPO_BATCH)
                if ds is not None:
                    ds = ds.to(DEVICE); da = da.to(DEVICE)
                    dl, _ = fwd(model, ds)
                    dagger_il = nn.functional.cross_entropy(dl, da)
                    loss += dagger_coef * dagger_il
                    dagger_loss_val = dagger_il.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            tp+=pl.item(); tv+=vl.item(); te+=ent.item()
            tt+=loss.item(); tdg+=dagger_loss_val; nb+=1

        nb = max(1, nb)
        print(f"  PPO {epoch}/{PPO_EPOCHS} loss={tt/nb:.4f} pol={tp/nb:.4f} val={tv/nb:.4f} "
              f"ent={te/nb:.4f} dagger={tdg/nb:.4f}", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS — per-phase
# ═══════════════════════════════════════════════════════════════════════════════
def eval_box_farming(model, n=20):
    """Phase 0 eval: solo farming. Returns (avg_boxes, survival_rate)."""
    model.eval(); boxes_t = 0; survived = 0
    for gi in range(n):
        ms = _EVAL_SEEDS[gi % N_EVAL_MAPS]; cid = gi % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=ms); obs = env.reset()
        init_boxes = int(np.sum(obs["map"]==2))
        opps = {pid: StopAgent(pid) for pid in range(4) if pid != cid}
        done = False; step = 0
        while not done:
            if int(obs["players"][cid][2]) != 1: break
            st = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl_ = int(obs["players"][cid][3])
            lm  = lmask(obs["map"], obs["bombs"], pos, bl_)
            with torch.no_grad(): a, _, _, _ = sample_a(model, st, lm, stoch=False)
            acts = [0,0,0,0]; acts[cid] = a
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1
        boxes_t  += init_boxes - int(np.sum(obs["map"]==2))
        if int(obs["players"][cid][2]) == 1: survived += 1
    avg_boxes = boxes_t / max(1, n); surv = survived / max(1, n)
    print(f"  [BoxFarm Eval] avg_boxes={avg_boxes:.1f}/{init_boxes} survival={surv:.2%}", flush=True)
    return avg_boxes, surv

def eval_escape(model, n=20):
    """
    Phase 1 eval: agent must place bombs and survive.
    Survival rate + fraction of safe bomb placements.
    """
    model.eval(); survived = 0; safe_bombs = 0; total_bombs = 0
    for gi in range(n):
        ms = _EVAL_SEEDS[gi % N_EVAL_MAPS]; cid = gi % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=ms); obs = env.reset()
        opps = {pid: StopAgent(pid) for pid in range(4) if pid != cid}
        done = False; step = 0
        while not done:
            if int(obs["players"][cid][2]) != 1: break
            st = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl_ = int(obs["players"][cid][3])
            lm  = lmask(obs["map"], obs["bombs"], pos, bl_)
            # Use shield mask in eval — measures POLICY quality, not exploration
            sm  = smask(obs["map"], obs["players"], obs["bombs"], cid, lm)
            with torch.no_grad(): a, _, _, _ = sample_a(model, st, sm, stoch=False)
            if a == 5:
                total_bombs += 1
                if safe_b(obs["map"], obs["players"], obs["bombs"], cid, pos, False):
                    safe_bombs += 1
            acts = [0,0,0,0]; acts[cid] = a
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1
        if int(obs["players"][cid][2]) == 1: survived += 1
    surv       = survived / max(1, n)
    safe_rate  = safe_bombs / max(1, total_bombs)
    print(f"  [Escape Eval] survival={surv:.2%} safe_bomb_rate={safe_rate:.2%} "
          f"({safe_bombs}/{total_bombs})", flush=True)
    return surv, safe_rate

def eval_1v1(model, opponent_cls, n=30, label="1v1"):
    """Phase 2/3 eval: 1v1 — 2 stop dummies, 1 opponent. Returns win rate."""
    model.eval(); wins = draws = losses = 0
    for gi in range(n):
        ms  = _EVAL_SEEDS[gi % N_EVAL_MAPS]; cid = gi % 4
        opp_id = [p for p in range(4) if p != cid][gi % 3]
        env = BomberEnv(max_steps=MAX_STEPS, seed=ms); obs = env.reset()
        opps = {pid: StopAgent(pid) for pid in range(4) if pid != cid}
        if opponent_cls is not None: opps[opp_id] = opponent_cls(opp_id)
        done = False; step = 0
        while not done:
            if int(obs["players"][cid][2]) != 1: break
            st = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl_ = int(obs["players"][cid][3])
            lm  = lmask(obs["map"], obs["bombs"], pos, bl_)
            sm  = smask(obs["map"], obs["players"], obs["bombs"], cid, lm)
            with torch.no_grad(): a, _, _, _ = sample_a(model, st, sm, stoch=False)
            pe  = sum(int(obs["players"][i][2]) for i in range(4) if i != cid)
            acts = [0,0,0,0]; acts[cid] = a
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1
        alive = [int(p[2]) for p in obs["players"]]
        if alive[cid] == 1 and alive[opp_id] == 0: wins += 1
        elif alive[cid] == alive[opp_id]: draws += 1
        else: losses += 1
    wr = wins / max(1, n)
    print(f"  [{label} Eval] W={wins} D={draws} L={losses} | WR={wr:.2%}", flush=True)
    return wr

def eval_full(model, n=30, label="FullGame"):
    """Phase 4 eval: full 4-player game vs mixed strong baselines."""
    model.eval(); wins = draws = losses = 0
    for gi in range(n):
        ms  = _EVAL_SEEDS[gi % N_EVAL_MAPS]; cid = gi % 4
        env = BomberEnv(max_steps=MAX_STEPS, seed=ms); obs = env.reset()
        opps = make_eval_opps(cid, ms + gi * 9_999_991)
        done = False; step = 0
        while not done:
            if int(obs["players"][cid][2]) != 1: break
            st = encode_obs(obs["map"], obs["players"], obs["bombs"], cid, step).unsqueeze(0).to(DEVICE)
            pos = (int(obs["players"][cid][0]), int(obs["players"][cid][1]))
            bl_ = int(obs["players"][cid][3])
            lm  = lmask(obs["map"], obs["bombs"], pos, bl_)
            sm  = smask(obs["map"], obs["players"], obs["bombs"], cid, lm)
            with torch.no_grad(): a, _, _, _ = sample_a(model, st, sm, stoch=False)
            acts = [0,0,0,0]; acts[cid] = a
            for pid, ag in opps.items(): acts[pid] = int(ag.act(obs))
            obs, terminated, truncated = env.step(acts)
            done = bool(terminated or truncated); step += 1
        alive = [int(p[2]) for p in obs["players"]]
        if alive[cid] == 1 and sum(alive) == 1: wins += 1
        elif alive[cid] == 1: draws += 1
        else: losses += 1
    wr = wins / max(1, n)
    print(f"  [{label} Eval] W={wins} D={draws} L={losses} | WR={wr:.2%}", flush=True)
    return wr

# ═══════════════════════════════════════════════════════════════════════════════
# MASTERY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
class MasteryTracker:
    """
    Rolling window of recent eval scores.
    Agent is "master" when ALL last `window` evals pass the threshold.
    This prevents noisy single-eval decisions from triggering premature advancement.
    """
    def __init__(self, window=MASTERY_WINDOW):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, passed: bool):
        self.history.append(passed)

    def is_master(self) -> bool:
        """True only if ALL recent evals passed."""
        return len(self.history) == self.window and all(self.history)

    def reset(self):
        self.history.clear()

# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM STATE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class CurriculumState:
    phase:          int   = PHASE_BOX
    round_in_phase: int   = 0   # rounds spent in current phase
    total_rounds:   int   = 0   # global round counter
    dagger_beta:    float = DAGGER_BETA_INIT
    dagger_coef:    float = DAGGER_COEF_INIT
    dagger_active:  bool  = False   # True once MAX_ROUNDS_PER_PHASE exceeded without mastery
    best_wins:      int   = -1

    def advance_phase(self):
        print(f"\n{'='*60}", flush=True)
        print(f"  ✅ MASTERED Phase {self.phase} ({PHASE_NAMES[self.phase]})", flush=True)
        print(f"  ➡️  Advancing to Phase {self.phase+1} ({PHASE_NAMES.get(self.phase+1,'Done')})", flush=True)
        print(f"{'='*60}\n", flush=True)
        self.phase          += 1
        self.round_in_phase  = 0
        self.dagger_beta     = DAGGER_BETA_INIT   # reset DAgger for new phase
        self.dagger_coef     = DAGGER_COEF_INIT
        self.dagger_active   = False

    def decay_dagger(self):
        """Called when agent passes mastery eval — reduce teacher reliance."""
        self.dagger_beta = max(DAGGER_BETA_MIN, self.dagger_beta * DAGGER_DECAY)
        self.dagger_coef = max(DAGGER_COEF_MIN, self.dagger_coef * DAGGER_DECAY)

    def activate_dagger(self):
        """Called when agent exceeds phase budget without mastery."""
        if not self.dagger_active:
            print(f"  ⚠️  DAgger ACTIVATED for phase {PHASE_NAMES[self.phase]} "
                  f"(beta={self.dagger_beta:.2f} coef={self.dagger_coef:.2f})", flush=True)
            self.dagger_active = True

    def phase_mastered_check(self, model, n_eval=20):
        """
        Run phase-appropriate eval and return (passed_bool, metric_dict).
        """
        if self.phase == PHASE_BOX:
            avg_boxes, surv = eval_box_farming(model, n=n_eval)
            passed = (avg_boxes >= BOX_MASTERY_THRESH) and (surv >= BOX_SURVIVAL_THRESH)
            return passed, {"avg_boxes": avg_boxes, "survival": surv}

        if self.phase == PHASE_ESCAPE:
            surv, safe_rate = eval_escape(model, n=n_eval)
            passed = (surv >= ESCAPE_SURVIVAL_THRESH) and (safe_rate >= SAFE_BOMB_THRESH)
            return passed, {"survival": surv, "safe_rate": safe_rate}

        if self.phase == PHASE_1V1_EASY:
            wr = eval_1v1(model, SimpleRuleAgent, n=n_eval, label="1v1-Simple")
            passed = wr >= WINRATE_1V1_SIMPLE
            return passed, {"win_rate": wr}

        if self.phase == PHASE_1V1_HARD:
            wr = eval_1v1(model, SmarterRuleAgent, n=n_eval, label="1v1-Smarter")
            passed = wr >= WINRATE_1V1_SMARTER
            return passed, {"win_rate": wr}

        # PHASE_FULL: no mastery threshold — run until budget exhausted
        wr = eval_full(model, n=n_eval, label="FullGame")
        passed = wr >= WINRATE_FULL
        return passed, {"win_rate": wr}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Train maps: {N_TRAIN_MAPS} fixed seeds | Eval maps: {N_EVAL_MAPS} held-out", flush=True)
    print(f"Max rounds: {MAX_TOTAL_ROUNDS} | Max per phase: {MAX_ROUNDS_PER_PHASE}", flush=True)
    print(f"DAgger beta={DAGGER_BETA_INIT} coef={DAGGER_COEF_INIT} decay={DAGGER_DECAY}", flush=True)
    print(f"\nMastery thresholds:", flush=True)
    print(f"  Phase 0 (Box):     boxes≥{BOX_MASTERY_THRESH} survival≥{BOX_SURVIVAL_THRESH:.0%}", flush=True)
    print(f"  Phase 1 (Escape):  survival≥{ESCAPE_SURVIVAL_THRESH:.0%} safe_bomb≥{SAFE_BOMB_THRESH:.0%}", flush=True)
    print(f"  Phase 2 (1v1 Easy): WR≥{WINRATE_1V1_SIMPLE:.0%}", flush=True)
    print(f"  Phase 3 (1v1 Hard): WR≥{WINRATE_1V1_SMARTER:.0%}", flush=True)
    print(f"  Phase 4 (Full):     WR≥{WINRATE_FULL:.0%} (or budget exhausted)\n", flush=True)

    model = BomberNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BomberNet: {n_params:,} parameters\n", flush=True)

    # Attempt to resume
    for path in [MODEL_PATH, BEST_PATH]:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                print(f"Resumed from {path}", flush=True); break
            except Exception as e:
                print(f"Could not load {path}: {e}", flush=True)

    # BC anchor (optional)
    bc_ds = BCDataset(BC_TRAIN_DIR); bc_loader = None
    if len(bc_ds) > 0:
        bc_loader = DataLoader(bc_ds, batch_size=min(128, PPO_BATCH), shuffle=False,
                               num_workers=0, drop_last=True)
        print(f"BC anchor: {len(bc_ds)} samples", flush=True)

    # Persistent optimizer + league
    optimizer = optim.AdamW(model.parameters(), lr=PPO_LR, weight_decay=WD)
    league    = League(LEAGUE_SIZE); league.add(model)
    ent       = ENT_INIT

    # Curriculum state + DAgger buffer
    cs             = CurriculumState()
    dagger_buffer  = DAggerBuffer(DAGGER_BUFFER_MAX)
    mastery_track  = MasteryTracker(MASTERY_WINDOW)

    # Initial diagnostics
    print("--- Initial diagnostics ---", flush=True)
    eval_box_farming(model, n=5)

    print(f"\n=== Skill-Gated Curriculum: up to {MAX_TOTAL_ROUNDS} total rounds ===\n", flush=True)

    while cs.total_rounds < MAX_TOTAL_ROUNDS:
        phase_name = PHASE_NAMES[cs.phase]
        use_dagger = cs.dagger_active or (cs.round_in_phase == 0)  # always start with some teacher
        use_bc     = (bc_loader is not None) and (cs.phase >= PHASE_1V1_EASY)
        frozen     = copy.deepcopy(model).cpu().eval()

        # DAgger coef only active from Phase 0 onward (all phases have teacher)
        active_dagger_coef = cs.dagger_coef if cs.dagger_active else (DAGGER_COEF_INIT * 0.5)
        active_dagger_beta = cs.dagger_beta if cs.dagger_active else (DAGGER_BETA_INIT * 0.3)
        # In Phase FULL, no teacher → no DAgger
        if cs.phase == PHASE_FULL:
            active_dagger_coef = 0.0; active_dagger_beta = 0.0

        print(f"\n--- Round {cs.total_rounds+1} | Phase={phase_name} "
              f"(rnd {cs.round_in_phase+1}) | ent={ent:.4f} "
              f"| dagger_beta={active_dagger_beta:.3f} coef={active_dagger_coef:.3f} ---",
              flush=True)

        safe_bomb_counter = [0, 0]
        rollouts = collect(
            model, frozen, GAMES_PER_ROUND, cs.phase, league,
            dagger_buffer=dagger_buffer if active_dagger_beta > 0 else None,
            dagger_beta=active_dagger_beta,
            safe_bomb_counter=safe_bomb_counter,
        )
        total_steps = sum(len(e.states) for e in rollouts)
        avg_reward  = np.mean([r for ep in rollouts for r in ep.rewards]) if rollouts else 0.0
        print(f"  collected {len(rollouts)} eps ({total_steps} steps) | avg_reward={avg_reward:.3f}", flush=True)
        if safe_bomb_counter[1] > 0:
            sb_rate = safe_bomb_counter[0] / safe_bomb_counter[1]
            print(f"  safe_bomb_rate (rollout) = {sb_rate:.2%} "
                  f"({safe_bomb_counter[0]}/{safe_bomb_counter[1]})", flush=True)

        ppo_update(
            model, rollouts, optimizer,
            bc_loader=bc_loader if use_bc else None,
            ent_coef=ent,
            dagger_buffer=dagger_buffer if active_dagger_coef > 0 and len(dagger_buffer) > 0 else None,
            dagger_coef=active_dagger_coef,
        )
        league.add(model)
        ent = max(ENT_MIN, ent * ENT_DECAY)

        # ── Mastery evaluation ────────────────────────────────────────────────
        passed, metrics = cs.phase_mastered_check(model, n_eval=20)
        mastery_track.update(passed)
        print(f"  Mastery eval: {'✅ PASS' if passed else '❌ FAIL'} | metrics={metrics}", flush=True)

        # ── Save best checkpoint for current phase ────────────────────────────
        phase_best = f"model_best_phase{cs.phase}.pth"
        # Use primary metric for "best"
        primary = metrics.get("win_rate", metrics.get("avg_boxes", metrics.get("survival", 0.0)))
        phase_key = f"_best_phase{cs.phase}"
        prev_best = getattr(cs, phase_key, -1.0)
        if primary > prev_best:
            setattr(cs, phase_key, primary)
            torch.save(model.state_dict(), phase_best)
            print(f"  ★ New phase best: {primary:.3f} → {phase_best}", flush=True)

        # ── Check overall best (full eval) for final checkpoint ───────────────
        if cs.phase == PHASE_FULL:
            wins_full = metrics.get("win_rate", 0.0) * 20
            if int(wins_full) > cs.best_wins:
                cs.best_wins = int(wins_full)
                torch.save(model.state_dict(), BEST_PATH)
                print(f"  ★ New overall best: {int(wins_full)}/20 → {BEST_PATH}", flush=True)

        # ── Curriculum advancement logic ───────────────────────────────────────
        cs.round_in_phase += 1
        cs.total_rounds   += 1

        if mastery_track.is_master():
            # Agent has consistently passed — advance phase
            cs.decay_dagger()   # decay teacher before advancing
            dagger_buffer.clear()   # fresh buffer for new phase
            mastery_track.reset()

            if cs.phase < PHASE_FULL:
                cs.advance_phase()
            else:
                print("  🏆 Phase FULL mastered! Continuing self-play...", flush=True)

        elif cs.round_in_phase >= MAX_ROUNDS_PER_PHASE and not cs.dagger_active:
            # Exceeded phase budget without mastery → activate DAgger
            cs.activate_dagger()

        # ── Hard cap per phase: force advance even if not mastered ────────────
        #    (prevents getting stuck forever in one phase)
        if cs.round_in_phase >= MAX_ROUNDS_PER_PHASE * 2 and cs.phase < PHASE_FULL:
            print(f"  ⏩ Force-advancing from {phase_name} (2x budget exceeded without mastery)", flush=True)
            dagger_buffer.clear(); mastery_track.reset()
            cs.advance_phase()

    # ─── Final eval ───────────────────────────────────────────────────────────
    print("\n=== Final evaluation (50 games) ===", flush=True)
    eval_full(model, n=50, label="Final-Current")

    if os.path.exists(BEST_PATH):
        print("\n=== Best checkpoint eval (50 games) ===", flush=True)
        best = BomberNet().to(DEVICE)
        best.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
        eval_full(best, n=50, label="Final-Best")

    print("Done.", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSION AGENT (drop-in inference wrapper)
# ═══════════════════════════════════════════════════════════════════════════════
class Agent:
    """
    Submission-ready agent.
    Loads the best checkpoint if available, otherwise the latest checkpoint.
    Uses the same encoding and shield mask as training/eval.
    """
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step = 0
        self.device = torch.device('cpu')
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.model = BomberNet().to(self.device)
        ckpt = BEST_PATH if os.path.exists(BEST_PATH) else MODEL_PATH
        if os.path.exists(ckpt):
            try:
                state = torch.load(ckpt, map_location=self.device)
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                self.model.load_state_dict(state)
            except Exception:
                pass
        self.model.eval()

    def act(self, obs: dict) -> int:
        try:
            players = obs['players']
            if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
                self.step += 1
                return 0

            st = encode_obs(obs['map'], players, obs['bombs'], self.agent_id, self.step).unsqueeze(0).to(self.device)
            pos = (int(players[self.agent_id][0]), int(players[self.agent_id][1]))
            bl_ = int(players[self.agent_id][3])
            lm = lmask(obs['map'], obs['bombs'], pos, bl_)
            sm = smask(obs['map'], players, obs['bombs'], self.agent_id, lm)

            with torch.no_grad():
                action, _, _, _ = sample_a(self.model, st, sm, stoch=False)
            self.step += 1
            return int(action)
        except Exception:
            self.step += 1
            return 0


if __name__ == "__main__":
    main()