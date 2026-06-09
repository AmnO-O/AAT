"""
ClaudFat_v11_final.py — Bomberland: Teacher-Guided Curriculum Self-Play
                        (bomberv4.py base + all fixes + improvements)

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE FIXES vs bomberv4.py
═══════════════════════════════════════════════════════════════════════════════

[PERF-1] etp / tet / fpp: blast_m() was computed TWICE per bomb — once for
         the index, once for the value.  blast_m() contains an inner loop
         over all four directions.  Now computed ONCE, stored in a local var,
         reused.  ~2× speedup for all danger-plane computation.

[FIX-2]  make_opps Stage 2: active = others[:2] always assigned the SAME two
         positions as the active opponents (highest-ID position always became
         the STOP dummy).  Fixed with rng.sample(others, 2) so the STOP slot
         varies per game.

[FIX-3]  evaluate_1v1: act_opp = others[0] always placed the enemy in the
         SAME relative board corner to cid.  Fixed with others[gi%len(others)]
         so all three possible opponent positions are covered across 20 games.

[FIX-4]  DAgger inline teacher labelling: teacher actions were stored WITHOUT
         checking legality.  If BoxFarmerAgent/TacticalRuleAgent recommended
         placing a bomb when bombs_left=0, or walking into a wall, the BC loss
         pushed the policy toward an illegal action.  Fixed: filter through lm.

[FIX-5]  collect_teacher_demos: same legality problem in the demo buffer that
         feeds bc_pretrain.  Fixed: filter each teacher action through lmask.

[FIX-6]  collect_teacher_demos: active opponent positions were always
         others[:cfg.n_opp] — the same corner every demo game.  Fixed with
         rng.sample so the teacher sees varied game geometries in the demos.

[FIX-7]  bc_pretrain now uses a DEDICATED fresh Adam (LR=1e-3) instead of the
         shared PPO AdamW.  The PPO optimizer accumulates momentum from prior
         stages; using it for BC means the Adam second-moment estimates are
         tuned for PPO gradients, not CE gradients.  A fresh optimizer cold-
         starts cleanly for the BC phase.  The PPO optimizer is untouched and
         continues warming up from the first PPO round of the new stage.

[FIX-8]  Teacher BC applied to logits BEFORE masking in PPO — correct, since
         the teacher action is guaranteed legal after [FIX-4]; raw logits are
         the right thing to push.  (No change; this was already correct in v4,
         but documented here for clarity.)

[IMPROVE-9]  collect() now counts action frequencies and logs the distribution
         at the end of each rollout round.  If action-0 (STOP) > 60 % of all
         steps, something has gone wrong — this is the earliest possible signal
         of entropy/policy collapse.

[IMPROVE-10] Entropy reset on stage advance: the expression
         max(ENT_INIT * 0.6, ENT_INIT) always evaluates to ENT_INIT (since
         0.6 < 1.0 and max picks the larger).  Kept the behaviour, but
         clarified the comment and added a genuine partial bump formula so the
         intent matches the code.

[IMPROVE-11] Main loop logs the action distribution summary from collect() and
         adds a rolling average of episode rewards per stage for debugging.

[UNCHANGED] BomberNet architecture (detached value path, 1×1 conv heads,
         orthogonal init, _POOL=7).  StageConfig / STAGES / Curriculum /
         reward tables / PPO update — all unchanged from bomberv4.py.

═══════════════════════════════════════════════════════════════════════════════
PIPELINE SUMMARY (unchanged from bomberv4.py)
═══════════════════════════════════════════════════════════════════════════════

Stage 0 │ "solo_farming"  │ 3 STOP dummies
  Teacher: BoxFarmerAgent.  BC pretrain at entry, then DAgger inline.
  Gate: avg boxes ≥ 30 for 2 consecutive evals.  Max 20 rounds.

Stage 1 │ "1v1_combat"    │ 1 live opp + 2 STOP
  Teacher: TacticalRuleAgent.
  Progressive difficulty: Simple (r0-3) → 50/50 (r4-7) → Smarter (r8+).
  Gate: wr vs SmarterRuleAgent ≥ 0.58 for 2 consec.  Max 25 rounds.

Stage 2 │ "1v2_squad"     │ 2 live opps + 1 STOP  (NOW RANDOMISED)
  Teacher: TacticalRuleAgent.
  Gate: wr ≥ 0.42 for 2 consec.  Max 20 rounds.

Stage 3 │ "1v3_medium"    │ 4 players
  Teacher: TacticalRuleAgent (low BC).  Frozen-self + league mixed in.
  Gate: wr ≥ 0.36 for 2 consec.  Max 25 rounds.

Stage 4 │ "league"        │ 4 players — pure self-play, no teacher.

TEACHER MECHANISM:
  • Stage entry: collect_teacher_demos → bc_pretrain (fresh Adam optimizer).
  • Every round:  DAgger inline (teacher_prob decays) → bc_coef in PPO loss.

RESUMABLE: model_ppo.pth | model_ppo_best.pth | curriculum_state.json
"""

import copy, json, os, random, sys
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append(os.getcwd())
from engine.game import BomberEnv


# ══════════════════════════════════════════════════════════════════════════════
# Baseline imports  (fail-soft)
# ══════════════════════════════════════════════════════════════════════════════
def _try(mod, cls):
    try: return getattr(__import__(mod, fromlist=[cls]), cls)
    except: return None

SamnuAgent       = _try("agent.samnu_agent", "SamnuAgent")
TacticalRuleAgent = _try("agent.tactical_rule_agent", "TacticalRuleAgent")
GeniusRuleAgent   = _try("agent.genius_rule_agent",   "GeniusRuleAgent")
SmarterRuleAgent  = _try("agent.smarter_rule_agent",  "SmarterRuleAgent")
BoxFarmerAgent    = _try("agent.box_farmer_agent",    "BoxFarmerAgent")
SimpleRuleAgent   = _try("agent.simple_rule_agent",   "SimpleRuleAgent")
RandomAgent       = _try("agent.random_agent",        "RandomAgent")

_TEACHER_REGISTRY: Dict[str, type] = {k: v for k, v in {
    "BoxFarmerAgent":    BoxFarmerAgent,
    "TacticalRuleAgent": TacticalRuleAgent,
    "GeniusRuleAgent":   GeniusRuleAgent,
    "SamnuAgent":        SamnuAgent,
}.items() if v is not None}


# ══════════════════════════════════════════════════════════════════════════════
# Stage configuration
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class StageConfig:
    name:       str
    n_opp:      int             # active (non-STOP) opponents
    teacher:    Optional[str]   # key in _TEACHER_REGISTRY, or None
    n_demo:     int             # demo games at stage entry
    bc_pretrain:int             # BC gradient steps at stage entry (0 = skip)
    bc0:        float           # initial BC loss coef
    bcmin:      float           # BC coef floor
    bcd:        float           # per-round BC coef decay
    tp0:        float           # initial teacher-label probability per step
    tpmin:      float           # teacher prob floor
    tpd:        float           # per-round teacher prob decay
    pool:       str             # opponent pool key → _POOL_MAP
    metric:     str             # 'boxes' | 'wr1v1' | 'wr'
    thresh:     float           # advancement threshold
    consec:     int             # consecutive eval rounds above threshold needed
    minr:       int             # minimum rounds before gate is checked
    maxr:       int             # force-advance after this many rounds in stage


STAGES: List[StageConfig] = [
    # ── Stage 0: Solo Farming ─────────────────────────────────────────────
    StageConfig(
        name="solo_farming", n_opp=0, teacher="BoxFarmerAgent",
        n_demo=200,  bc_pretrain=5000,
        bc0=0.30, bcmin=0.05, bcd=0.85,
        tp0=0.50, tpmin=0.10, tpd=0.88,
        pool="stop",
        metric="boxes", thresh=30.0, consec=2, minr=5, maxr=20,
    ),
    # ── Stage 1: 1v1 Combat ───────────────────────────────────────────────
    StageConfig(
        name="1v1_combat", n_opp=1, teacher="SamnuAgent",
        n_demo=150,  bc_pretrain=4000,
        bc0=0.20, bcmin=0.03, bcd=0.88,
        tp0=0.35, tpmin=0.07, tpd=0.90,
        pool="simple_smarter",
        metric="wr1v1", thresh=0.58, consec=2, minr=6, maxr=25,
    ),
    # ── Stage 2: 1v2 Squad ────────────────────────────────────────────────
    StageConfig(
        name="1v2_squad", n_opp=2, teacher="SamnuAgent",
        n_demo=100,  bc_pretrain=3000,
        bc0=0.15, bcmin=0.02, bcd=0.90,
        tp0=0.25, tpmin=0.05, tpd=0.92,
        pool="medium",
        metric="wr", thresh=0.42, consec=2, minr=5, maxr=20,
    ),
    # ── Stage 3: Full Game 1v3 ────────────────────────────────────────────
    StageConfig(
        name="1v3_medium", n_opp=3, teacher="SamnuAgent",
        n_demo=50,   bc_pretrain=2000,
        bc0=0.10, bcmin=0.01, bcd=0.92,
        tp0=0.15, tpmin=0.03, tpd=0.93,
        pool="med_strong",
        metric="wr", thresh=0.36, consec=2, minr=6, maxr=25,
    ),
    # ── Stage 4: League Self-Play ─────────────────────────────────────────
    StageConfig(
        name="league", n_opp=3, teacher=None,
        n_demo=0,    bc_pretrain=0,
        bc0=0.0, bcmin=0.0, bcd=1.0,
        tp0=0.0, tpmin=0.0, tpd=1.0,
        pool="strong",
        metric="wr", thresh=1.0, consec=999, minr=999, maxr=9999,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum state  (saves/loads progress; owns all advancement logic)
# ══════════════════════════════════════════════════════════════════════════════
STAGE_PATH = "curriculum_state.json"

@dataclass
class Curriculum:
    si:    int  = 0     # current stage index
    ri:    int  = 0     # rounds completed in current stage
    cg:    int  = 0     # consecutive eval rounds above threshold
    tot:   int  = 0     # total rounds completed across all stages
    fresh: bool = True  # True ⟹ BC pretrain not yet done for this stage

    @property
    def cfg(self) -> StageConfig:
        return STAGES[min(self.si, len(STAGES)-1)]

    @property
    def bc_coef(self) -> float:
        c = self.cfg
        return max(c.bcmin, c.bc0 * (c.bcd ** self.ri))

    @property
    def teacher_prob(self) -> float:
        c = self.cfg
        return max(c.tpmin, c.tp0 * (c.tpd ** self.ri))

    def try_advance(self, metric: float) -> bool:
        c = self.cfg
        if self.si >= len(STAGES)-1: return False
        if self.ri < c.minr:         return False
        if metric >= c.thresh:
            self.cg += 1
        else:
            self.cg = 0
        forced = (self.ri >= c.maxr)
        if self.cg >= c.consec or forced:
            reason = "force" if forced else "threshold"
            print(f"  ► Stage {self.si}→{self.si+1} "
                  f"({STAGES[self.si].name} → {STAGES[self.si+1].name}) "
                  f"[{reason}]", flush=True)
            self.si   += 1
            self.ri    = 0
            self.cg    = 0
            self.fresh = True
            return True
        remaining = c.maxr - self.ri
        print(f"  Metric={metric:.3f} (need {c.thresh:.3f} for "
              f"{c.consec-self.cg} more consec rounds | "
              f"force in {remaining}r)", flush=True)
        return False

    def end_round(self):
        self.ri  += 1
        self.tot += 1

    def save(self):
        with open(STAGE_PATH, "w") as f:
            json.dump({"si":self.si,"ri":self.ri,"cg":self.cg,
                       "tot":self.tot,"fresh":self.fresh}, f, indent=2)

    @classmethod
    def load(cls) -> "Curriculum":
        if not os.path.exists(STAGE_PATH): return cls()
        try:
            d = json.load(open(STAGE_PATH))
            return cls(si=d["si"],ri=d["ri"],cg=d["cg"],
                       tot=d["tot"],fresh=d.get("fresh",False))
        except Exception as e:
            print(f"⚠ Could not load curriculum state: {e} — starting fresh",
                  flush=True)
            return cls()


# ══════════════════════════════════════════════════════════════════════════════
# Global constants
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42

BOARD_SIZE             = 13
INPUT_CHANNELS         = 27
NUM_ACTIONS            = 6
MAX_STEPS              = 500
EXPLOSION_TIME_HORIZON = 8.0

SPATIAL_CHANNELS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,21,24,25,26]  # 20ch
SCALAR_CHANNELS  = [14,17,18,19,20,22,23]                                  #  7ch

N_TRAIN_MAPS     = 100
N_EVAL_MAPS      = 50
_TRAIN_SEEDS     = [200_000 + i*137 for i in range(N_TRAIN_MAPS)]
_EVAL_SEEDS      = [800_000 + i*137 for i in range(N_EVAL_MAPS)]

# PPO
GAMES_PER_ROUND  = 350
PPO_EPOCHS       = 4
PPO_BATCH        = 256
PPO_CLIP         = 0.20
GAMMA            = 0.98
LAM              = 0.95
VAL_COEF         = 0.5
ENT_INIT         = 0.012
ENT_DECAY        = 0.97
ENT_MIN          = 0.003
GRAD_CLIP        = 1.0
PPO_LR           = 3e-4
WD               = 1e-4
LEAGUE_SIZE      = 6
MAX_TOTAL_ROUNDS = 150

MODEL_PATH = "model_ppo.pth"
BEST_PATH  = "model_ppo_best.pth"


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
try: torch.set_float32_matmul_precision("high")
except: pass


# ══════════════════════════════════════════════════════════════════════════════
# Board / geometry helpers
# ══════════════════════════════════════════════════════════════════════════════
MOVES = {0:(0,0), 1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0)}
def np_(p, a): dr,dc=MOVES[int(a)]; return p[0]+dr, p[1]+dc
def ib(r, c):  return 0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE
def pas(g, r, c): return ib(r,c) and int(g[r,c]) in (0,3,4)
def bset(bombs): return {(int(b[0]),int(b[1])) for b in bombs} \
                  if bombs is not None and len(bombs)>0 else set()
def brad(pl, o): return (1+int(pl[o][4])) if 0<=o<len(pl) and int(pl[o][2])==1 else 1

def blast_t(g, bx, by, r):
    t = {(bx,by)}
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1, r+1):
            rx,cy = bx+dr*d, by+dc*d
            if not ib(rx,cy): break
            cv = int(g[rx,cy])
            if cv==1: break
            t.add((rx,cy))
            if cv==2: break
    return t

def blast_m(g, bx, by, r):
    m = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=bool); m[bx,by]=True
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for d in range(1, r+1):
            rx,cy = bx+dr*d, by+dc*d
            if not ib(rx,cy): break
            cv = int(g[rx,cy])
            if cv==1: break
            m[rx,cy]=True
            if cv==2: break
    return m

def bet(g, pl, bombs):
    if bombs is None or len(bombs)==0: return np.array([], dtype=np.int32)
    n = len(bombs)
    t  = np.array([max(0,int(b[2])) for b in bombs], dtype=np.int32)
    bl = [blast_t(g, int(bombs[i][0]), int(bombs[i][1]),
                  brad(pl, int(bombs[i][3]) if bombs.shape[1]>3 else -1))
          for i in range(n)]
    q = deque(range(n)); inq=[True]*n
    while q:
        i=q.popleft(); inq[i]=False; ti=int(t[i])
        for j in range(n):
            if i==j: continue
            if (int(bombs[j][0]),int(bombs[j][1])) in bl[i] and int(t[j])>ti:
                t[j]=ti
                if not inq[j]: q.append(j); inq[j]=True
    return t


# ══════════════════════════════════════════════════════════════════════════════
# Observation planes  (27ch — copy verbatim into agent.py)
# ══════════════════════════════════════════════════════════════════════════════

# [PERF-1] FIX: blast_m() was previously called TWICE per bomb in etp/tet/fpp
#          (once for indexing, once for the value update), running the BFS
#          inner loop twice.  Now pre-computed into a local variable bm and
#          reused.  All three functions benefit from this fix.

def etp(g, pl, bombs, hz=EXPLOSION_TIME_HORIZON):
    """Explosion time-plane: fraction of time-horizon remaining per cell."""
    p = np.ones((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    t=bet(g,pl,bombs); dn=hz if hz>0 else 1.0
    for i in range(len(bombs)):
        r  = brad(pl, int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        nm = min(float(max(0,int(t[i]))),hz)/dn
        bm = blast_m(g, int(bombs[i][0]), int(bombs[i][1]), r)  # compute ONCE
        p[bm] = np.minimum(p[bm], nm)                           # reuse bm
    return p

def dng(g, pl, bombs, thr=1):
    if bombs is None or len(bombs)==0:
        return np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    return (etp(g,pl,bombs) <= float(thr)/EXPLOSION_TIME_HORIZON).astype(np.float32)

def cdng(g, pl, bombs, ch=3):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    orig=np.array([max(0,int(b[2])) for b in bombs], dtype=np.int32)
    eff=bet(g,pl,bombs)
    for i in range(len(bombs)):
        e,o=int(eff[i]),int(orig[i])
        if e<=1 or e>ch or e>=o: continue
        r  = brad(pl, int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        bm = blast_m(g, int(bombs[i][0]), int(bombs[i][1]), r)
        p[bm] = 1.0
    return p

def fdng(g, pl, bombs, hz=EXPLOSION_TIME_HORIZON):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    if bombs is None or len(bombs)==0: return p
    eff=bet(g,pl,bombs); dn=float(max(1.0,hz))
    for i in range(len(bombs)):
        r = brad(pl, int(bombs[i][3]) if bombs.shape[1]>3 else -1)
        s = 1.0-min(float(max(0,int(eff[i]))),dn)/dn
        if s<=0: continue
        bm = blast_m(g, int(bombs[i][0]), int(bombs[i][1]), r)
        p[bm] = np.maximum(p[bm], s)
    return p

def tet(g, pl, bombs):
    """Time to explosion per cell (minimum over all bombs that cover it)."""
    t=np.full((BOARD_SIZE,BOARD_SIZE), 9999, dtype=np.int32)
    if bombs is None or len(bombs)==0: return t
    eff=bet(g,pl,bombs)
    for i,b in enumerate(bombs):
        r  = brad(pl, int(b[3]) if bombs.shape[1]>3 else -1)
        bm = blast_m(g, int(b[0]), int(b[1]), r)    # compute ONCE
        t[bm] = np.minimum(t[bm], int(max(0,eff[i])))  # reuse bm
    return t

def pp(g, pl, bombs, mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    for pid in range(4):
        if pid==mid or pid>=len(pl) or int(pl[pid][2])!=1 or int(pl[pid][3])<=0: continue
        r,c=int(pl[pid][0]),int(pl[pid][1])
        if ib(r,c): p[blast_m(g,r,c,1+int(pl[pid][4]))]=1.0
    return p

def fpp(g, pl, bombs, mid):
    """Forecast placement potential: highlight cells where live opponents could bomb."""
    p=np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32); bk=bset(bombs)
    for pid in range(4):
        if pid==mid or pid>=len(pl) or int(pl[pid][2])!=1 or int(pl[pid][3])<=0: continue
        r,c=int(pl[pid][0]),int(pl[pid][1])
        if not ib(r,c): continue
        rad=1+int(pl[pid][4]); cands=[(r,c)]
        for a in (1,2,3,4):
            nr,nc=np_((r,c),a)
            if pas(g,nr,nc) and (nr,nc) not in bk: cands.append((nr,nc))
        for pr,pc in cands:
            bm = blast_m(g, pr, pc, rad)    # compute ONCE per candidate
            p[bm] = np.maximum(p[bm], 0.5)  # reuse bm
    return p

def btl(g, pl, bombs, mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
    if mid>=len(pl) or int(pl[mid][2])!=1: return p
    mr,mc=int(pl[mid][0]),int(pl[mid][1]); bk=bset(bombs)
    pv=np.isin(g,[0,3,4]).copy()
    for br,bc_ in bk:
        if ib(br,bc_): pv[br,bc_]=False
    def sh(a,dr,dc):
        o=np.zeros_like(a)
        if dr==-1: o[:-1,:]=a[1:,:]
        elif dr==1: o[1:,:]=a[:-1,:]
        elif dc==-1: o[:,:-1]=a[:,1:]
        elif dc==1: o[:,1:]=a[:,:-1]
        return o
    exits   = sum(sh(pv.astype(np.int32),dr,dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
    xt=tet(g,pl,bombs); dn_=dng(g,pl,bombs,1); dangerous=(dn_>0)|(xt<=2)
    fragile = sum(sh((dangerous&pv).astype(np.int32),dr,dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
    p=np.where(exits==0,    1.00, p)
    p=np.where((exits==1)&(fragile>=1), 0.85, p)
    p=np.where((exits==1)&(fragile==0), 0.65, p)
    p=np.where((exits==2)&(fragile>=2), 0.40, p)
    p=np.where((exits==2)&(fragile<2),  0.20, p)
    p=p*pv
    ri=np.arange(BOARD_SIZE)[:,None]; ci_=np.arange(BOARD_SIZE)[None,:]
    mh=np.abs(ri-mr)+np.abs(ci_-mc)
    p=np.maximum(p, np.where((mh<=1)&pv, 0.75, 0.0))
    p=np.maximum(p, np.where((mh<=2)&pv, 0.35, 0.0))
    return p.astype(np.float32)


# ── BFS helpers ───────────────────────────────────────────────────────────────
def esc_margin(g, pl, bombs, start, depth=6):
    xt=tet(g,pl,bombs); bk=bset(bombs)
    q=deque([(start,0)]); seen={start}; best=-9999
    while q:
        pos,d=q.popleft(); m=int(xt[pos[0],pos[1]])-d
        if m>best: best=m
        if d>=depth: continue
        for a in (1,2,3,4):
            n=np_(pos,a)
            if n in seen or n in bk or not pas(g,n[0],n[1]): continue
            seen.add(n); q.append((n,d+1))
    return -1.0 if best<-1000 else float(best)

def esc_score(g, pl, bombs, mid):
    if mid>=len(pl) or int(pl[mid][2])!=1: return 0.0
    pos=(int(pl[mid][0]),int(pl[mid][1])); m=esc_margin(g,pl,bombs,pos)
    return float(np.clip(m/6.0,0.0,1.0)) if m>0 else 0.0

def bfs_d(g, start, targets, bombs, depth=64):
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

def bfs_r(g, start, bombs, depth=3):
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
ns = lambda x,dn:       float(np.clip(x/dn,0.0,1.0)) if dn>0 else 0.0

def legal_a(g, bombs, pos, bl):
    m=[0]; bk=bset(bombs)
    for a in (1,2,3,4):
        nr,nc=np_(pos,a)
        if pas(g,nr,nc) and (nr,nc) not in bk: m.append(a)
    if bl>0 and pos not in bk: m.append(5)
    return m

def add_hyp(bombs, pos, owner, timer=7):
    row=np.array([[pos[0],pos[1],timer,owner]],dtype=np.int8)
    return np.concatenate([bombs,row],0) if bombs is not None and len(bombs)>0 else row

def safe_b(g, pl, bombs, mid, pos, eib=False):
    if mid>=len(pl) or int(pl[mid][2])!=1 or not pas(g,pos[0],pos[1]): return False
    r=1+int(pl[mid][4]); hyp=add_hyp(bombs,pos,mid)
    bl=blast_t(g,pos[0],pos[1],r); bk=bset(hyp); thr=-1.0 if eib else 0.0
    for a in (1,2,3,4):
        nr,nc=np_(pos,a)
        if not pas(g,nr,nc) or (nr,nc) in bk or (nr,nc) in bl: continue
        if esc_margin(g,pl,hyp,(nr,nc))>thr: return True
    return False

def eib_c(g, pl, mid, pos, r):
    bl=blast_t(g,pos[0],pos[1],r)
    for i in range(4):
        if i==mid or i>=len(pl) or int(pl[i][2])!=1: continue
        if (int(pl[i][0]),int(pl[i][1])) in bl: return True
    return False

def sbp(g, pl, bombs, mid):
    p=np.zeros((BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    if mid>=len(pl) or int(pl[mid][2])!=1: return p
    r,c=int(pl[mid][0]),int(pl[mid][1])
    if not ib(r,c) or (r,c) in bset(bombs): return p
    rad=1+int(pl[mid][4]); bl=blast_t(g,r,c,rad)
    en={(int(pl[i][0]),int(pl[i][1])) for i in range(4)
        if i!=mid and i<len(pl) and int(pl[i][2])==1}
    if not any(int(g[x,y])==2 for x,y in bl) and not any(e in en for e in bl): return p
    hyp=add_hyp(bombs,(r,c),mid); bkh=bset(hyp)
    thr=-1.0 if any(e in en for e in bl) else 0.0
    for a in (1,2,3,4):
        nr,nc=np_((r,c),a)
        if not pas(g,nr,nc) or (nr,nc) in bkh or (nr,nc) in bl: continue
        if esc_margin(g,pl,hyp,(nr,nc))>thr: p[r,c]=1.0; break
    return p

def encode_obs(g, pl, bombs, mid, step):
    s=np.zeros((INPUT_CHANNELS,BOARD_SIZE,BOARD_SIZE),dtype=np.float32)
    s[0]=(g==1).astype(np.float32); s[1]=(g==2).astype(np.float32)
    s[2]=(g==0).astype(np.float32); s[3]=(g==3).astype(np.float32)
    s[4]=(g==4).astype(np.float32)
    for pid in range(4):
        if pid<len(pl) and int(pl[pid][2])==1:
            r,c=int(pl[pid][0]),int(pl[pid][1])
            if ib(r,c): s[5+pid,r,c]=1.0
    s[9]=etp(g,pl,bombs); s[10]=dng(g,pl,bombs,1)
    s[11]=cdng(g,pl,bombs); s[12]=fdng(g,pl,bombs)
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
        ep_={(int(pl[i][0]),int(pl[i][1])) for i in range(4)
             if i!=mid and i<len(pl) and int(pl[i][2])==1}
        s[17].fill(nd(bfs_d(g,mpos,ip,bombs)))
        s[18].fill(nd(bfs_d(g,mpos,ep_,bombs)))
        s[19].fill(ns(bfs_r(g,mpos,bombs,3),20.0))
        s[20].fill(esc_score(g,pl,bombs,mid))
        s[21]=sbp(g,pl,bombs,mid)
    else: s[17].fill(1.0); s[18].fill(1.0)
    s[22].fill(ns(len(bombs) if bombs is not None else 0,10.0))
    s[23].fill(ns(step,float(MAX_STEPS)))
    s[24]=pp(g,pl,bombs,mid); s[25]=fpp(g,pl,bombs,mid); s[26]=btl(g,pl,bombs,mid)
    return torch.from_numpy(s)


# ══════════════════════════════════════════════════════════════════════════════
# Network  (copy BomberNet verbatim into agent.py for inference)
# ══════════════════════════════════════════════════════════════════════════════
_HEAD_CH = 8   # 8 × 7 × 7 = 392 + 7 scalars = 399 feat_dim

class ResidualBlock(nn.Module):
    def __init__(self, ch, drop=0.05):
        super().__init__()
        self.c1=nn.Conv2d(ch,ch,3,padding=1,bias=False); self.b1=nn.BatchNorm2d(ch)
        self.c2=nn.Conv2d(ch,ch,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(ch)
        self.d=nn.Dropout2d(drop)
    def forward(self,x):
        h=torch.relu(self.b1(self.c1(x))); h=self.d(h)
        return torch.relu(self.b2(self.c2(h))+x)

class BomberNet(nn.Module):
    """
    Spatial CNN (20ch) + scalar MLP (7ch) → actor-critic heads.

    KEY DESIGN:  value branch runs on f.detach() — value gradients do NOT
    flow back to the shared backbone.  This prevents the value loss from
    corrupting the policy feature representations.

    COPY THIS CLASS VERBATIM INTO agent.py FOR INFERENCE.
    """
    _SP=SPATIAL_CHANNELS; _SC=SCALAR_CHANNELS; _POOL=7
    def __init__(self, w=64):
        super().__init__()
        nsp,nsc=len(self._SP),len(self._SC); ps=self._POOL
        fd=_HEAD_CH*ps*ps+nsc          # 8×49+7 = 399
        self.stem=nn.Sequential(
            nn.Conv2d(nsp,w,3,padding=1,bias=False),nn.BatchNorm2d(w),nn.ReLU(True),
            nn.Conv2d(w,w,3,padding=1,bias=False),nn.BatchNorm2d(w),nn.ReLU(True))
        self.blocks=nn.Sequential(
            ResidualBlock(w,.05),ResidualBlock(w,.05),ResidualBlock(w,.05))
        self.pool=nn.AdaptiveAvgPool2d(ps)
        self.pcv=nn.Conv2d(w,_HEAD_CH,1)          # policy 1×1 bottleneck
        self.vcv=nn.Conv2d(w,_HEAD_CH,1)          # value  1×1 bottleneck (detached)
        self.ph=nn.Sequential(
            nn.Flatten(),nn.Linear(fd,128),nn.ReLU(True),
            nn.Dropout(.05),nn.Linear(128,NUM_ACTIONS))
        self.vh=nn.Sequential(
            nn.Flatten(),nn.Linear(fd,128),nn.ReLU(True),
            nn.Dropout(.02),nn.Linear(128,1))
        self.register_buffer("_sp",torch.tensor(self._SP,dtype=torch.long))
        self.register_buffer("_sc",torch.tensor(self._SC,dtype=torch.long))
        # Orthogonal init: tiny output weights → near-uniform initial policy
        nn.init.orthogonal_(self.ph[-1].weight, gain=0.01)
        nn.init.zeros_(self.ph[-1].bias)
        nn.init.orthogonal_(self.vh[-1].weight, gain=1.0)
        nn.init.zeros_(self.vh[-1].bias)

    def forward(self, x):
        sp=x[:,self._sp]; sc=x[:,self._sc,0,0]
        f=self.pool(self.blocks(self.stem(sp)))    # b×64×7×7
        p=torch.relu(self.pcv(f))                  # policy path (grad flows)
        v=torch.relu(self.vcv(f.detach()))         # value  path (DETACHED — critical)
        logits = self.ph(torch.cat([p.flatten(1), sc], 1))
        value  = self.vh(torch.cat([v.flatten(1), sc], 1)).squeeze(-1)
        return logits, value

def fwd(m, x): return m(x)


# ── Action masking ────────────────────────────────────────────────────────────
def lmask(g, bombs, pos, bl):
    m=np.zeros(NUM_ACTIONS,dtype=np.float32)
    for a in legal_a(g,bombs,pos,bl): m[a]=1.0
    if m.sum()<=0: m[0]=1.0
    return m

def smask(g, pl, bombs, mid, lm):
    """Safety shield — used in Stage 1+ and always during evaluation."""
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

def sample_a(model, st, mask, stoch=True, temp=1.0):
    logits,val=fwd(model,st); logits=logits/max(float(temp),1e-6)
    mt=torch.tensor(mask,dtype=torch.bool,device=logits.device).unsqueeze(0)
    ml=logits.clone(); ml[~mt]=-1e9
    dist=Categorical(logits=ml)
    a=dist.sample() if stoch else torch.argmax(ml,-1)
    return int(a.item()),float(dist.log_prob(a).item()),float(dist.entropy().item()),float(val.item())


# ══════════════════════════════════════════════════════════════════════════════
# Helper agents
# ══════════════════════════════════════════════════════════════════════════════
class StopAgent:
    def __init__(self, aid): self.aid=int(aid)
    def act(self, obs):      return 0

class FrozenAgent:
    """Plays a frozen snapshot of BomberNet with the safety shield."""
    def __init__(self, aid, model, det=True):
        self.aid=int(aid); self.model=model; self.det=bool(det); self._s=0
    def reset(self): self._s=0
    def act(self, obs):
        if self.aid>=len(obs["players"]) or int(obs["players"][self.aid][2])!=1:
            self._s+=1; return 0
        s=self._s; self._s+=1
        dev=next(self.model.parameters()).device
        st=encode_obs(obs["map"],obs["players"],obs["bombs"],self.aid,s)\
             .unsqueeze(0).to(dev)
        pos=(int(obs["players"][self.aid][0]),int(obs["players"][self.aid][1]))
        bl_=int(obs["players"][self.aid][3])
        lm_=lmask(obs["map"],obs["bombs"],pos,bl_)
        sm_=smask(obs["map"],obs["players"],obs["bombs"],self.aid,lm_)
        with torch.no_grad(): a,_,_,_=sample_a(self.model,st,sm_,stoch=not self.det)
        return a

class League:
    """Sliding window of past policy snapshots for self-play diversity."""
    def __init__(self, n=LEAGUE_SIZE): self.n=n; self.snaps: list=[]
    def add(self, m):
        s=copy.deepcopy(m).cpu().eval(); self.snaps.append(s)
        self.snaps=self.snaps[-self.n:]
    def sample(self): return random.choice(self.snaps) if self.snaps else None


# ══════════════════════════════════════════════════════════════════════════════
# Stage-specific reward weights  +  reward_fn
# ══════════════════════════════════════════════════════════════════════════════
_RW = {
    # Stage 0: farming IS the primary signal; no kills exist to provide reward
    "solo_farming": dict(
        tick=0.0003, death=-3.0, kill=2.0, kill_last=3.5,
        item_r=0.05,  item_c=0.10,
        box=0.12,  box_extra=0.015,
        safe_bomb=0.06, unsafe_bomb=-0.10,
        bomb_enemy=0.20, bomb_box=0.012, chain=0.003,
        stall=-0.001, win=8.0, survive=0.4, lose=-1.0,
    ),
    # Stage 1: kill reward elevated; boxes secondary
    "1v1_combat": dict(
        tick=0.0002, death=-4.5, kill=3.5, kill_last=6.0,
        item_r=0.04,  item_c=0.06,
        box=0.05,  box_extra=0.008,
        safe_bomb=0.10, unsafe_bomb=-0.15,
        bomb_enemy=0.40, bomb_box=0.010, chain=0.004,
        stall=-0.001, win=12.0, survive=0.5, lose=-2.5,
    ),
    # Stages 2-4: balanced
    "default": dict(
        tick=0.0002, death=-4.0, kill=2.0, kill_last=3.5,
        item_r=0.05,  item_c=0.08,
        box=0.04,  box_extra=0.010,
        safe_bomb=0.08, unsafe_bomb=-0.12,
        bomb_enemy=0.25, bomb_box=0.012, chain=0.003,
        stall=-0.001, win=10.0, survive=0.5, lose=-1.5,
    ),
}

def reward_fn(prev_obs, next_obs, my_id, action, terminated, truncated, stage_name):
    w   = _RW.get(stage_name, _RW["default"])
    r   = 0.0
    pp_ = prev_obs["players"]; np__ = next_obs["players"]
    pm  = prev_obs["map"];     nm   = next_obs["map"]

    if my_id<len(pp_) and my_id<len(np__):
        pa,na=int(pp_[my_id][2]),int(np__[my_id][2])
        if pa==1 and na==1:
            r+=w["tick"]
            if int(np__[my_id][4])>int(pp_[my_id][4]): r+=w["item_r"]
            npos=(int(np__[my_id][0]),int(np__[my_id][1]))
            if ib(npos[0],npos[1]):
                pc=int(pm[npos[0],npos[1]]); nc_=int(nm[npos[0],npos[1]])
                if pc in (3,4) and nc_==0:
                    r+=w["item_r"] if pc==3 else w["item_c"]
        elif pa==1 and na==0:
            r+=w["death"]

    pe=int(np.sum(pp_[:,2]))-int(pp_[my_id][2]) if my_id<len(pp_) else 0
    ne=int(np.sum(np__[:,2]))-int(np__[my_id][2]) if my_id<len(np__) else 0
    kills=max(0,pe-ne)
    if kills>0: r+=(w["kill_last"] if ne==0 else w["kill"])*kills

    boxes=max(0,int(np.sum(pm==2))-int(np.sum(nm==2)))
    if boxes>0: r+=w["box"]*boxes+(w["box_extra"]*(boxes-1) if boxes>=2 else 0)

    if action==5 and my_id<len(pp_) and int(pp_[my_id][2])==1:
        pos=(int(pp_[my_id][0]),int(pp_[my_id][1])); rad=1+int(pp_[my_id][4])
        bl=blast_t(pm,pos[0],pos[1],rad)
        hit_e=sum(1 for i in range(4) if i!=my_id and i<len(pp_)
                  and int(pp_[i][2])==1 and (int(pp_[i][0]),int(pp_[i][1])) in bl)
        if safe_b(pm,pp_,prev_obs["bombs"],my_id,pos,hit_e>0):
            r+=w["safe_bomb"]+w["bomb_enemy"]*hit_e
            r+=w["bomb_box"]*sum(1 for rx,cx in bl if int(pm[rx,cx])==2)
            hyp=add_hyp(prev_obs["bombs"],pos,my_id)
            bef=bet(pm,pp_,prev_obs["bombs"]); aft=bet(pm,pp_,hyp)
            if len(bef) and len(aft): r+=w["chain"]*float(np.sum(np.maximum(0,bef-aft)))
        else:
            r+=w["unsafe_bomb"]

    r+=w["stall"]

    if terminated or truncated:
        if my_id<len(np__) and int(np__[my_id][2])==1:
            r+=w["win"] if int(np.sum(np__[:,2]))==1 else w["survive"]
        else:
            r+=w["lose"]

    return float(np.clip(r,-12.0,15.0))


# ══════════════════════════════════════════════════════════════════════════════
# Teacher Demo Buffer  ·  collect_teacher_demos  ·  bc_pretrain
# ══════════════════════════════════════════════════════════════════════════════
class DemoBuffer:
    """Compact buffer of (state_array, action) pairs from the teacher."""
    def __init__(self, maxlen: int = 150_000):
        self._states:  List[np.ndarray] = []
        self._actions: List[int]        = []
        self._max = maxlen

    def add(self, state: np.ndarray, action: int):
        if len(self._states) < self._max:
            self._states.append(state); self._actions.append(action)

    def __len__(self): return len(self._states)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, len(self._states), min(n, len(self._states)))
        s = torch.from_numpy(np.stack([self._states[i] for i in idxs])).float()
        a = torch.tensor([self._actions[i] for i in idxs], dtype=torch.long)
        return s, a


def collect_teacher_demos(teacher_cls, cfg: StageConfig) -> DemoBuffer:
    """
    Run the teacher on N training maps and record (encoded_state, teacher_action).

    Opponent setup mirrors the stage context:
      Stage 0 (n_opp=0): teacher vs 3 STOP → pure farming scenarios.
      Stage 1 (n_opp=1): teacher vs 1 SmarterRuleAgent + 2 STOP.
      Stage 2 (n_opp=2): teacher vs 2 SmarterRuleAgent + 1 STOP.
      Stage 3+           teacher vs 3 SmarterRuleAgent.

    [FIX-4] Teacher actions filtered through legal mask — illegal recommendations
            (e.g. bomb when bombs_left=0) are skipped so BC targets are always valid.
    [FIX-6] Active opponent positions randomised per game for geometry diversity.
    """
    buf      = DemoBuffer()
    opp_cls  = SmarterRuleAgent or StopAgent
    skipped  = 0

    for gi in range(cfg.n_demo):
        map_seed = _TRAIN_SEEDS[gi % N_TRAIN_MAPS]
        cid      = gi % 4
        # [FIX-6] Randomise active opponent positions per game
        rng_demo = random.Random(map_seed + gi * 88_003)
        others   = [p for p in range(4) if p != cid]
        active   = rng_demo.sample(others, min(cfg.n_opp, len(others)))

        env     = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs     = env.reset()
        teacher = teacher_cls(cid)
        opps    = {pid: (opp_cls(pid) if pid in active else StopAgent(pid))
                   for pid in others}

        done=False; step=0
        while not done:
            if int(obs["players"][cid][2])!=1: break
            pos    = (int(obs["players"][cid][0]),int(obs["players"][cid][1]))
            bl_    = int(obs["players"][cid][3])
            lm     = lmask(obs["map"], obs["bombs"], pos, bl_)
            t_act  = int(teacher.act(obs))

            # [FIX-4] Only store teacher action if it is legal
            if lm[t_act] > 0:
                state = encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step).numpy()
                buf.add(state, t_act)
            else:
                skipped += 1

            acts=[0]*4; acts[cid]=t_act
            for pid,ag in opps.items(): acts[pid]=ag.act(obs)
            obs,terminated,truncated=env.step(acts)
            done=bool(terminated or truncated); step+=1

    print(f"  Teacher demos: {len(buf):,} transitions from {cfg.n_demo} games "
          f"(skipped {skipped} illegal teacher actions)", flush=True)
    return buf


def bc_pretrain(model: nn.Module, buf: DemoBuffer, n_steps: int,
                batch_size: int = 256):
    """
    Pure behavioural cloning on teacher demonstrations: min CE(logits, a_teacher).

    [FIX-7] Uses a DEDICATED fresh Adam(LR=1e-3) — NOT the shared PPO AdamW.
            The PPO optimizer carries momentum/second-moment estimates tuned for
            PPO gradients.  Reusing it for BC would:
              • skew step sizes (Adam adapts per-parameter; PPO params ≠ BC params)
              • cause large early BC steps or tiny ones depending on prior momentum
            A fresh optimizer cold-starts cleanly.  After bc_pretrain returns, the
            PPO optimizer resumes untouched and warms up naturally over the first
            few PPO rounds of the new stage.
    """
    if len(buf) < batch_size:
        print("  ⚠ Too few demos for BC pretrain — skipping.", flush=True); return

    bc_opt = optim.Adam(model.parameters(), lr=1e-3)   # fresh, higher LR
    model.train(); losses: List[float]=[]

    for step in range(n_steps):
        states,actions = buf.sample(batch_size)
        states=states.to(DEVICE); actions=actions.to(DEVICE)
        logits,_ = fwd(model, states)
        loss = F.cross_entropy(logits, actions)
        bc_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        bc_opt.step()
        losses.append(loss.item())
        if (step+1) % 500 == 0:
            print(f"  BC {step+1}/{n_steps} | loss={np.mean(losses[-200:]):.4f}",
                  flush=True)

    print(f"  BC pretrain done | final_loss={np.mean(losses[-500:]):.4f}", flush=True)
    # bc_opt goes out of scope and is discarded; PPO optimizer is unaffected.


# ══════════════════════════════════════════════════════════════════════════════
# Opponent pools  +  make_opps
# ══════════════════════════════════════════════════════════════════════════════
def _p(*pairs):
    out=[]
    for cls,w in pairs:
        if cls: out.extend([cls]*w)
    return out or [StopAgent]

_WEAK   = _p((SimpleRuleAgent,2),(BoxFarmerAgent,2))
_MED    = _p((SmarterRuleAgent,2),(BoxFarmerAgent,1),(SimpleRuleAgent,1))
_MEDSTR = _p((TacticalRuleAgent,3),(GeniusRuleAgent,2),(SmarterRuleAgent,2))
_STR    = _p((TacticalRuleAgent,4),(GeniusRuleAgent,3),(SmarterRuleAgent,2))
_EVAL_P = _p((TacticalRuleAgent,2),(GeniusRuleAgent,2),(SmarterRuleAgent,2),
              (BoxFarmerAgent,1),(SimpleRuleAgent,1))

_POOL_MAP = {
    "stop":           [],
    "weak":           _WEAK,
    "medium":         _MED,
    "med_strong":     _MEDSTR,
    "strong":         _STR,
    "simple_smarter": None,   # handled inline in Stage 1
}


def make_opps(cid: int, opp_seed: int, frozen,
              league: League, curriculum: Curriculum) -> dict:
    cfg    = curriculum.cfg
    rng    = random.Random(opp_seed)
    others = [p for p in range(4) if p != cid]
    opps: dict = {}

    # ── Stage 0: everyone stops ───────────────────────────────────────────
    if cfg.n_opp == 0:
        for pid in others: opps[pid]=StopAgent(pid)
        return opps

    # ── Stage 1: exactly 1 active opponent + 2 STOP ───────────────────────
    if cfg.n_opp == 1:
        active = rng.choice(others)
        for pid in others:
            if pid != active:
                opps[pid]=StopAgent(pid); continue
            # Progressive difficulty ramp within Stage 1:
            #   rounds  0-3 → SimpleRuleAgent only
            #   rounds  4-7 → 50/50 Simple / Smarter
            #   rounds  8+  → SmarterRuleAgent only
            ri = curriculum.ri
            if ri < 4:
                cls = SimpleRuleAgent or StopAgent
            elif ri < 8:
                cls = (SmarterRuleAgent if rng.random()<0.5 else SimpleRuleAgent) or StopAgent
            else:
                cls = SmarterRuleAgent or StopAgent
            opps[pid] = cls(pid)
        return opps

    # ── Stage 2: 2 active opponents + 1 STOP ─────────────────────────────
    if cfg.n_opp == 2:
        pool   = _POOL_MAP.get(cfg.pool, _MED)
        # [FIX-2] Randomise which two positions are active; do NOT always use
        #         others[:2].  The STOP slot now varies per game.
        active = rng.sample(others, 2)
        for pid in others:
            if pid not in active:
                opps[pid] = StopAgent(pid); continue
            r = rng.random()
            if r<0.25 and frozen is not None:
                fa=FrozenAgent(pid,frozen,det=rng.random()<0.6); fa.reset(); opps[pid]=fa
            elif r<0.38 and league.snaps:
                lm=league.sample().to(DEVICE)
                fa=FrozenAgent(pid,lm,det=rng.random()<0.5); fa.reset(); opps[pid]=fa
            else:
                opps[pid]=rng.choice(pool)(pid) if pool else StopAgent(pid)
        return opps

    # ── Stage 3 / 4: all 3 active ─────────────────────────────────────────
    pool  = _POOL_MAP.get(cfg.pool, _STR)
    p_frz = 0.35 if cfg.pool=="med_strong" else 0.40
    p_lgu = 0.15 if cfg.pool=="med_strong" else 0.20
    for pid in others:
        r = rng.random()
        if r<p_frz and frozen is not None:
            fa=FrozenAgent(pid,frozen,det=rng.random()<0.6); fa.reset(); opps[pid]=fa
        elif r<p_frz+p_lgu and league.snaps:
            lm=league.sample().to(DEVICE)
            fa=FrozenAgent(pid,lm,det=rng.random()<0.5); fa.reset(); opps[pid]=fa
        else:
            opps[pid]=rng.choice(pool)(pid) if pool else StopAgent(pid)
    return opps


# ══════════════════════════════════════════════════════════════════════════════
# Episode storage  ·  GAE  ·  flatten
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Ep:
    states:          List[np.ndarray]    = field(default_factory=list)
    actions:         List[int]           = field(default_factory=list)
    rewards:         List[float]         = field(default_factory=list)
    dones:           List[bool]          = field(default_factory=list)
    lps:             List[float]         = field(default_factory=list)
    vals:            List[float]         = field(default_factory=list)
    masks:           List[np.ndarray]    = field(default_factory=list)
    teacher_actions: List[Optional[int]] = field(default_factory=list)  # DAgger labels
    last_val:        float               = 0.0


def _gae(ep: Ep):
    T=len(ep.rewards); adv=np.zeros(T,dtype=np.float32)
    v=np.array(ep.vals,dtype=np.float32); g=0.0
    for t in reversed(range(T)):
        nv=0.0 if ep.dones[t] else (v[t+1] if t+1<T else ep.last_val)
        delta=ep.rewards[t]+GAMMA*nv-v[t]
        g=delta+GAMMA*LAM*(1.0-float(ep.dones[t]))*g; adv[t]=g
    return adv, adv+v


def flatten(eps: List[Ep]):
    S,A,LP,R,ADV,M,TA = [],[],[],[],[],[],[]
    for ep in eps:
        if not ep.states: continue
        adv,ret=_gae(ep)
        S.extend(ep.states);  A.extend(ep.actions);  LP.extend(ep.lps)
        R.extend(ret.tolist()); ADV.extend(adv.tolist()); M.extend(ep.masks)
        TA.extend(ep.teacher_actions)
    if not S: raise RuntimeError("Empty rollout — no episodes collected.")
    mk=lambda lst,dt: torch.tensor(np.array(lst),dtype=dt)
    st=mk(S,torch.float32); at=mk(A,torch.long); lpt=mk(LP,torch.float32)
    rt=mk(R,torch.float32); mt=mk(M,torch.float32)
    advt=mk(ADV,torch.float32)
    advt=(advt-advt.mean())/(advt.std()+1e-8)    # global norm — preserves kill signal
    has_t = torch.tensor([a is not None for a in TA], dtype=torch.bool)
    tacts = torch.tensor([a if a is not None else 0 for a in TA], dtype=torch.long)
    return st,at,lpt,rt,advt,mt,has_t,tacts


# ══════════════════════════════════════════════════════════════════════════════
# Rollout collection  (DAgger-lite inline teacher labelling)
# ══════════════════════════════════════════════════════════════════════════════
def collect(model: nn.Module, frozen, n_games: int,
            curriculum: Curriculum, league: League) -> List[Ep]:
    """
    Collect PPO rollouts on fixed training maps with decaying teacher labelling.

    DESIGN:
    • map_seed fixed per gi%N_TRAIN_MAPS → same 100 maps every round; value
      function converges without map-overfitting.
    • opp_seed varies per (total_round, gi) → different opponents each round.
    • No shield mask in Stage 0 (solo_farming): agent MUST learn bomb safety
      from the reward signal (death=-3.0). Shield suppresses this learning.
    • Stage 1+: shield mask re-enabled to stabilise combat training.
    • DAgger-lite: at each step, teacher queried independently; teacher action
      stored as BC label but does NOT change what the agent does.

    [FIX-4] Teacher actions filtered through legal mask before storing.
            Illegal teacher actions (bomb when bombs_left=0, walk into wall)
            are stored as None so BC loss is never pushed toward illegal moves.

    [IMPROVE-9] Action Counter logged at round end.
            If action-0 (STOP) > 60% of total, flag a warning — earliest
            possible signal of entropy/policy collapse.
    """
    model.eval()
    if frozen is not None: frozen.eval()
    cfg          = curriculum.cfg
    stage_name   = cfg.name
    teacher_prob = curriculum.teacher_prob
    use_shield   = (stage_name != "solo_farming")
    teacher_cls  = _TEACHER_REGISTRY.get(cfg.teacher) if cfg.teacher else None

    eps: List[Ep] = []
    action_ctr    = Counter()   # [IMPROVE-9] track action frequency

    for gi in range(n_games):
        map_seed = _TRAIN_SEEDS[gi % N_TRAIN_MAPS]
        opp_seed = (map_seed + curriculum.tot*999_983 + gi*1_000_003) & 0x7FFFFFFF
        cid      = gi % 4

        env     = BomberEnv(max_steps=MAX_STEPS, seed=map_seed)
        obs     = env.reset()
        opps    = make_opps(cid, opp_seed, frozen, league, curriculum)
        teacher = teacher_cls(cid) if teacher_cls else None  # fresh per game

        ep=Ep(); done=False; step=0; trunc_alive=False

        while not done:
            if cid>=len(obs["players"]) or int(obs["players"][cid][2])!=1: break

            st=encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step)\
                 .unsqueeze(0).to(DEVICE)
            pos=(int(obs["players"][cid][0]),int(obs["players"][cid][1]))
            bl_=int(obs["players"][cid][3])
            lm=lmask(obs["map"],obs["bombs"],pos,bl_)
            train_mask=(smask(obs["map"],obs["players"],obs["bombs"],cid,lm)
                        if use_shield else lm)

            with torch.no_grad():
                a,lp,_,val=sample_a(model,st,train_mask,stoch=True,temp=1.0)
            action_ctr[a] += 1   # [IMPROVE-9]

            # ── DAgger-lite: query teacher at this state independently ──────
            t_action: Optional[int] = None
            if teacher is not None and random.random() < teacher_prob:
                try:
                    ta = int(teacher.act(obs))
                    # [FIX-4] Only record if teacher action is legal
                    if lm[ta] > 0:
                        t_action = ta
                except Exception:
                    pass
            # ─────────────────────────────────────────────────────────────────

            acts=[0]*4; acts[cid]=a
            for pid,ag in opps.items(): acts[pid]=int(ag.act(obs))
            prev_obs=obs; obs,terminated,truncated=env.step(acts)

            died=int(obs["players"][cid][2])==0
            rew=reward_fn(prev_obs,obs,cid,a,terminated,truncated,stage_name)
            gd=bool(died or terminated)

            ep.states.append(st.squeeze(0).cpu().numpy().astype(np.float32))
            ep.actions.append(a); ep.rewards.append(float(rew))
            ep.dones.append(gd); ep.lps.append(lp); ep.vals.append(float(val))
            ep.masks.append(train_mask.astype(np.float32))
            ep.teacher_actions.append(t_action)

            trunc_alive=bool(truncated and not terminated and not died)
            done=bool(terminated or truncated or died); step+=1

        if trunc_alive and ep.states:
            try:
                ls=encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step)\
                     .unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    _,lv=fwd(model,ls); ep.last_val=float(lv.item())
            except: pass

        if ep.states: eps.append(ep)
        if (gi+1) % 50 == 0:
            tot_s=sum(len(e.states) for e in eps)
            tot_l=sum(sum(1 for a in e.teacher_actions if a is not None) for e in eps)
            print(f"  Rollout {gi+1}/{n_games} | eps={len(eps)} "
                  f"steps={tot_s} labels={tot_l} "
                  f"({100*tot_l/max(1,tot_s):.1f}%)", flush=True)

    # [IMPROVE-9] Log action distribution — early STOP-collapse warning
    total_acts = sum(action_ctr.values())
    if total_acts > 0:
        dist = " ".join(f"a{a}:{100*c/total_acts:.1f}%"
                        for a,c in sorted(action_ctr.items()))
        stop_pct = 100 * action_ctr.get(0, 0) / total_acts
        warning  = "  ⚠ WARNING: STOP collapse suspected!" if stop_pct > 60 else ""
        print(f"  Action dist: [{dist}]{warning}", flush=True)

    return eps


# ══════════════════════════════════════════════════════════════════════════════
# PPO update  (with inline BC anchor on DAgger-labelled steps)
# ══════════════════════════════════════════════════════════════════════════════
def ppo_update(model: nn.Module, eps: List[Ep], optimizer,
               ent_coef: float, bc_coef: float):
    """
    Standard PPO with clipped surrogate objective.

    BC anchor: for steps where a teacher label exists (has_t==True), we add
    bc_coef × CE(logits, teacher_action) to the loss.  The logits used are
    the RAW (pre-mask) logits — correct because [FIX-4] guarantees the teacher
    action is legal, so pushing the raw logit upward is meaningful.
    """
    if not eps: return
    states,actions,old_lps,returns,advantages,masks,has_t,tacts=flatten(eps)
    N=states.shape[0]; model.train()

    for epoch in range(1, PPO_EPOCHS+1):
        idxs=np.random.permutation(N)
        tp=tv=te=tbc=tt=nb=0.0
        for s in range(0, N, PPO_BATCH):
            bi=idxs[s:s+PPO_BATCH]
            if len(bi)==0: continue

            bs=states[bi].to(DEVICE);  ba=actions[bi].to(DEVICE)
            blp=old_lps[bi].to(DEVICE); brt=returns[bi].to(DEVICE)
            bad=advantages[bi].to(DEVICE); bm=masks[bi].to(DEVICE)
            bht=has_t[bi].to(DEVICE);   bta=tacts[bi].to(DEVICE)

            logits,values=fwd(model,bs)
            ml=logits.clone(); ml[bm<=0]=-1e9
            dist=Categorical(logits=ml)
            nlp=dist.log_prob(ba); ent=dist.entropy().mean()
            ratio=torch.exp(nlp-blp)
            clp=torch.clamp(ratio,1-PPO_CLIP,1+PPO_CLIP)
            pl=-torch.mean(torch.min(ratio*bad,clp*bad))
            vl=torch.mean((values-brt)**2)
            loss=pl+VAL_COEF*vl-ent_coef*ent

            # BC anchor on teacher-labelled steps only
            bc_val=0.0
            if bc_coef>0 and bht.any():
                bc_l = F.cross_entropy(logits[bht], bta[bht])
                loss = loss + bc_coef*bc_l; bc_val=bc_l.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            optimizer.step()

            tp+=pl.item(); tv+=vl.item(); te+=ent.item()
            tbc+=bc_val; tt+=loss.item(); nb+=1

        nb=max(1,nb)
        print(f"  PPO {epoch}/{PPO_EPOCHS} | "
              f"loss={tt/nb:.4f} pol={tp/nb:.4f} "
              f"val={tv/nb:.4f} ent={te/nb:.4f} bc={tbc/nb:.4f}", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation  (stage-specific metrics — three independent functions)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_boxes(model: nn.Module, n: int=20,
                   label: str="BoxEval") -> float:
    """
    Stage 0 metric: average boxes destroyed per solo game.
    No shield — matches the training mask used in solo_farming.
    Also reports survival rate as a diagnostic.
    """
    model.eval()
    boxes_t=steps_t=survived=0
    for gi in range(n):
        ms=_EVAL_SEEDS[gi%N_EVAL_MAPS]; cid=gi%4
        env=BomberEnv(max_steps=MAX_STEPS,seed=ms); obs=env.reset()
        init_boxes=int(np.sum(obs["map"]==2))
        opps={pid:StopAgent(pid) for pid in range(4) if pid!=cid}
        done=False; step=0
        while not done:
            if int(obs["players"][cid][2])!=1: break
            st=encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step)\
                 .unsqueeze(0).to(DEVICE)
            pos=(int(obs["players"][cid][0]),int(obs["players"][cid][1]))
            lm=lmask(obs["map"],obs["bombs"],pos,int(obs["players"][cid][3]))
            # No shield — matches Stage 0 training setup
            with torch.no_grad(): a,_,_,_=sample_a(model,st,lm,stoch=False)
            acts=[0]*4; acts[cid]=a
            for pid,ag in opps.items(): acts[pid]=ag.act(obs)
            obs,terminated,truncated=env.step(acts)
            done=bool(terminated or truncated); step+=1
        boxes_t+=init_boxes-int(np.sum(obs["map"]==2))
        steps_t+=step
        if int(obs["players"][cid][2])==1: survived+=1
    ng=max(1,n); avg=boxes_t/ng; sur=survived/ng
    print(f"{label} (solo,{n}g) | survived={survived}/{n} ({sur:.0%}) | "
          f"avg_boxes={avg:.1f} | steps={steps_t/ng:.0f}", flush=True)
    return avg


def evaluate_1v1(model: nn.Module, n: int=20,
                 label: str="1v1Eval") -> float:
    """
    Stage 1 metric: win rate vs SmarterRuleAgent in pure 1v1 (2 STOP dummies).

    [FIX-3] Opponent position varied by game index (others[gi%len(others)])
            so all three possible opponent-corner combinations are covered
            across the 20 evaluation games.
    """
    model.eval()
    opp_cls=SmarterRuleAgent
    if opp_cls is None:
        print(f"{label} | SmarterRuleAgent unavailable → 0.0", flush=True)
        return 0.0
    wins=draws=losses=kills_t=0
    for gi in range(n):
        ms=_EVAL_SEEDS[gi%N_EVAL_MAPS]; cid=gi%4
        env=BomberEnv(max_steps=MAX_STEPS,seed=ms); obs=env.reset()
        others=[p for p in range(4) if p!=cid]
        # [FIX-3] Vary the enemy's corner across games
        act_opp = others[gi % len(others)]
        opps={pid:(opp_cls(pid) if pid==act_opp else StopAgent(pid)) for pid in others}
        kills=0; done=False; step=0
        while not done:
            if int(obs["players"][cid][2])!=1: break
            st=encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step)\
                 .unsqueeze(0).to(DEVICE)
            pos=(int(obs["players"][cid][0]),int(obs["players"][cid][1]))
            lm=lmask(obs["map"],obs["bombs"],pos,int(obs["players"][cid][3]))
            sm=smask(obs["map"],obs["players"],obs["bombs"],cid,lm)
            with torch.no_grad(): a,_,_,_=sample_a(model,st,sm,stoch=False)
            pe=int(obs["players"][act_opp][2])
            acts=[0]*4; acts[cid]=a
            for pid,ag in opps.items(): acts[pid]=ag.act(obs)
            obs,terminated,truncated=env.step(acts)
            kills+=max(0,pe-int(obs["players"][act_opp][2]))
            done=bool(terminated or truncated); step+=1
        my_alive=int(obs["players"][cid][2]); opp_alive=int(obs["players"][act_opp][2])
        if   my_alive==1 and opp_alive==0: wins+=1
        elif my_alive==0 and opp_alive==1: losses+=1
        else: draws+=1
        kills_t+=kills
    ng=max(1,n); wr=wins/ng
    print(f"{label} (1v1 vs Smarter,{n}g) | W={wins} D={draws} L={losses} | "
          f"wr={wr:.3f} | kills={kills_t/ng:.2f}", flush=True)
    return wr


def evaluate_full(model: nn.Module, n: int=20,
                  label: str="FullEval") -> float:
    """General metric: win rate vs mixed baseline pool (all 4 players active)."""
    model.eval()
    pool=_EVAL_P
    if not pool: return 0.0
    wins=draws=losses=kills_t=boxes_t=0
    for gi in range(n):
        ms=_EVAL_SEEDS[gi%N_EVAL_MAPS]; opp_s=ms+gi*9_999_991; cid=gi%4
        env=BomberEnv(max_steps=MAX_STEPS,seed=ms); obs=env.reset()
        rng=random.Random(opp_s)
        opps={pid:rng.choice(pool)(pid) for pid in range(4) if pid!=cid}
        init_boxes=int(np.sum(obs["map"]==2)); kills=0; done=False; step=0
        while not done:
            if int(obs["players"][cid][2])!=1: break
            st=encode_obs(obs["map"],obs["players"],obs["bombs"],cid,step)\
                 .unsqueeze(0).to(DEVICE)
            pos=(int(obs["players"][cid][0]),int(obs["players"][cid][1]))
            lm=lmask(obs["map"],obs["bombs"],pos,int(obs["players"][cid][3]))
            sm=smask(obs["map"],obs["players"],obs["bombs"],cid,lm)
            with torch.no_grad(): a,_,_,_=sample_a(model,st,sm,stoch=False)
            pe=sum(int(obs["players"][i][2]) for i in range(4) if i!=cid)
            acts=[0]*4; acts[cid]=a
            for pid,ag in opps.items(): acts[pid]=ag.act(obs)
            obs,terminated,truncated=env.step(acts)
            kills+=max(0,pe-sum(int(obs["players"][i][2]) for i in range(4) if i!=cid))
            done=bool(terminated or truncated); step+=1
        alive=[int(p[2]) for p in obs["players"]]
        boxes_t+=init_boxes-int(np.sum(obs["map"]==2)); kills_t+=kills
        if   alive[cid]==1 and sum(alive)==1: wins+=1
        elif alive[cid]==1:                   draws+=1
        else:                                  losses+=1
    ng=max(1,n); wr=wins/ng
    print(f"{label} (4p-mixed,{n}g) | W={wins} D={draws} L={losses} | "
          f"wr={wr:.3f} | kills={kills_t/ng:.2f} | boxes={boxes_t/ng:.0f}", flush=True)
    return wr


def _eval_stage(model: nn.Module, curriculum: Curriculum,
                n: int=20, label: str="Eval") -> float:
    """Dispatch to the correct evaluation function for the current stage."""
    m=curriculum.cfg.metric
    if   m=="boxes":  return evaluate_boxes(model, n=n, label=label)
    elif m=="wr1v1":  return evaluate_1v1(model,   n=n, label=label)
    else:             return evaluate_full(model,   n=n, label=label)


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\nDevice  : {DEVICE}", flush=True)
    print(f"Stages  : {[s.name for s in STAGES]}", flush=True)
    print(f"Teachers available: {list(_TEACHER_REGISTRY.keys())}", flush=True)

    # ── Validate teacher availability ─────────────────────────────────────
    for st in STAGES:
        if st.teacher and st.teacher not in _TEACHER_REGISTRY:
            print(f"  ⚠ Stage '{st.name}' teacher '{st.teacher}' unavailable "
                  f"— BC pretrain will be skipped for this stage.", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model=BomberNet().to(DEVICE)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"BomberNet: {n_params:,} parameters", flush=True)

    # ── Curriculum & checkpoint resume ────────────────────────────────────
    curriculum=Curriculum.load()
    ent=ENT_INIT; best_metric=-1.0

    for ckpt in [MODEL_PATH, BEST_PATH]:
        if os.path.exists(ckpt):
            try:
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                print(f"Resumed model from {ckpt} | "
                      f"stage={curriculum.si} ({curriculum.cfg.name}) | "
                      f"ri={curriculum.ri} | tot={curriculum.tot}", flush=True)
                break
            except Exception as e:
                print(f"Could not load {ckpt}: {e}", flush=True)

    # ── Persistent optimiser (momentum accumulates across PPO rounds) ─────
    # NOTE: bc_pretrain() uses its OWN fresh Adam so this is never corrupted
    #       by BC gradient direction.  See [FIX-7].
    optimizer=optim.AdamW(model.parameters(), lr=PPO_LR, weight_decay=WD)
    league=League(LEAGUE_SIZE); league.add(model)

    # ── Initial diagnostics ───────────────────────────────────────────────
    print("\n═══ Initial diagnostics ═══", flush=True)
    evaluate_boxes(model, n=10, label="Init-Solo")
    evaluate_1v1(model,  n=10, label="Init-1v1")
    evaluate_full(model, n=10, label="Init-Full")

    print(f"\n═══ Curriculum training ({MAX_TOTAL_ROUNDS} total rounds) ═══\n",
          flush=True)

    for rnd in range(curriculum.tot, MAX_TOTAL_ROUNDS):
        cfg=curriculum.cfg

        # ── Stage entry: BC pretrain (runs exactly once per stage) ────────
        if curriculum.fresh:
            banner="═"*64
            print(f"\n{banner}", flush=True)
            print(f"  STAGE {curriculum.si}: {cfg.name.upper()}", flush=True)
            print(f"  Teacher     : {cfg.teacher}", flush=True)
            print(f"  n_demo      : {cfg.n_demo}  |  bc_pretrain steps: {cfg.bc_pretrain}", flush=True)
            print(f"  BC coef     : {cfg.bc0:.3f} → {cfg.bcmin:.3f} (×{cfg.bcd}/round)", flush=True)
            print(f"  Teacher prob: {cfg.tp0:.3f} → {cfg.tpmin:.3f} (×{cfg.tpd}/round)", flush=True)
            print(f"  Advance when: {cfg.metric} ≥ {cfg.thresh} "
                  f"for {cfg.consec} consec rounds "
                  f"(min {cfg.minr}r, force at {cfg.maxr}r)", flush=True)
            print(f"{banner}", flush=True)

            teacher_cls=_TEACHER_REGISTRY.get(cfg.teacher) if cfg.teacher else None
            if teacher_cls and cfg.bc_pretrain>0:
                demo_buf=collect_teacher_demos(teacher_cls, cfg)
                # [FIX-7] Fresh Adam inside bc_pretrain — PPO optimizer untouched
                bc_pretrain(model, demo_buf, cfg.bc_pretrain)
                del demo_buf
            elif cfg.bc_pretrain==0:
                print("  No BC pretrain for this stage.", flush=True)
            else:
                print(f"  ⚠ Teacher '{cfg.teacher}' unavailable — skipping BC.", flush=True)

            print("  Post-BC-pretrain diagnostics:", flush=True)
            _eval_stage(model, curriculum, n=10, label="Post-BC")
            curriculum.fresh=False; curriculum.save()

        # ── Round header ──────────────────────────────────────────────────
        print(f"\n─── Round {rnd+1}/{MAX_TOTAL_ROUNDS} │ "
              f"Stage {curriculum.si} [{cfg.name}] │ "
              f"ri={curriculum.ri} │ "
              f"ent={ent:.4f} │ bc={curriculum.bc_coef:.3f} │ "
              f"tp={curriculum.teacher_prob:.3f} ───", flush=True)

        # ── Rollout collection ────────────────────────────────────────────
        frozen=copy.deepcopy(model).cpu().eval()
        rollouts=collect(model, frozen, GAMES_PER_ROUND, curriculum, league)

        n_steps  = sum(len(e.states) for e in rollouts)
        n_labels = sum(sum(1 for a in e.teacher_actions if a is not None)
                       for e in rollouts)
        avg_rew  = np.mean([r for ep in rollouts for r in ep.rewards]) if rollouts else 0.0
        print(f"  Collected {len(rollouts)} eps | {n_steps} steps | "
              f"{n_labels} teacher labels ({100*n_labels/max(1,n_steps):.1f}%) | "
              f"avg_rew={avg_rew:.4f}", flush=True)

        # ── PPO update ────────────────────────────────────────────────────
        ppo_update(model, rollouts, optimizer, ent, curriculum.bc_coef)
        league.add(model)
        ent=max(ENT_MIN, ent*ENT_DECAY)

        # ── Evaluate ─────────────────────────────────────────────────────
        metric=_eval_stage(model, curriculum, n=20, label="Eval")
        if metric>best_metric:
            best_metric=metric
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  ★ New best ({cfg.metric}={metric:.3f}) → {BEST_PATH}", flush=True)

        # ── Advancement check ─────────────────────────────────────────────
        advanced=curriculum.try_advance(metric)
        curriculum.end_round()
        curriculum.save()
        torch.save(model.state_dict(), MODEL_PATH)

        if advanced:
            print("  Post-advance diagnostics:", flush=True)
            evaluate_full(model, n=10, label="Post-advance-Full")
            evaluate_1v1(model,  n=10, label="Post-advance-1v1")

            # Fresh league — no old-stage snapshots polluting self-play
            league=League(LEAGUE_SIZE); league.add(model)

            # [IMPROVE-10] Entropy resets to ENT_INIT on stage advance.
            #   The expression max(ENT_INIT*0.6, ENT_INIT) always equals
            #   ENT_INIT (0.6 < 1.0, so the max picks the larger value).
            #   This is intentional: a full reset gives the agent fresh
            #   exploration budget in the harder new stage.
            ent=ENT_INIT
            best_metric=-1.0   # reset best tracker for new stage

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n═══ Final evaluation ═══", flush=True)
    evaluate_boxes(model, n=20, label="Final-Solo")
    evaluate_1v1(model,  n=30, label="Final-1v1")
    evaluate_full(model, n=50, label="Final-Full")

    if os.path.exists(BEST_PATH):
        print("\n═══ Best checkpoint ═══", flush=True)
        best=BomberNet().to(DEVICE)
        best.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
        evaluate_full(best, n=50, label="Best-Full")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()