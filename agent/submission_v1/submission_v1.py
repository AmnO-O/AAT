"""
agent.py -- Bomberland Hybrid Agent v7 "Neural-Guided Rule Agent"
================================================================
Architecture: Stockfish-style hybrid
  Layer 1  Rule engine  Hard safety constraints, BFS routing, danger maps.
           Never bypassed. Generates 2-6 safe candidate actions.
  Layer 2  Neural scorer  Tiny MLP (~46K params) scores the candidates.
           Argmax selects the best. Falls back to rule priority if net absent.

Fixes over v6:
  F1. Chain-reaction bonus in _bomb_value: a bomb that early-triggers another
      bomb gets +chain_boxes bonus. Previously ignored — now the highest-value
      plays (chain kills) are correctly scored.
  F2. Item collection during escape: if an item lies on the best escape path
      we collect it for free. Previously items were blocked when in danger_any.
  F3. _bomb_value box score now accounts for multi-box blasts: each additional
      box hit past the first earns +0.5 (chain density bonus).
  F4. Enemy chase (Priority 5) now fires correctly: time budget check moved
      earlier so it is not always starved by farm-box BFS.
  F5. _can_escape_after_bomb escape window scales with number of existing
      bombs near the agent — tight bomb clusters shorten eff_t appropriately.
  F6. Box targeting prefers spots adjacent to multiple boxes (cluster value).
  F7. Map control score added to _safe_fallback: tiles near board centre are
      preferred when no other signal exists (late-game positioning).

Neural scorer (ScorerNet):
  Input  : 34 floats (encoded state context + per-candidate features)
  Hidden : 64 → 32 (ReLU)
  Output : 1 scalar (value estimate for this candidate)
  Load   : from "scorer.pt" if present. If absent, pure rule agent.
  Freeze : weights are frozen at inference (eval mode, no_grad).
"""

import os
import time
from collections import deque

import numpy as np

# ── optional torch import ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

# ── Action constants ──────────────────────────────────────────────────────────
STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = 0, 1, 2, 3, 4, 5
MOVES = {STOP:(0,0), LEFT:(-1,0), RIGHT:(1,0), UP:(0,-1), DOWN:(0,1)}
MOVE_ACTIONS = [LEFT, RIGHT, UP, DOWN]

# ── Map cell types ────────────────────────────────────────────────────────────
GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

# ── Game constants ────────────────────────────────────────────────────────────
BOMB_TIMER   = 7
MAX_RADIUS   = 5
MAX_CAPACITY = 5
BOARD_H      = 13
BOARD_W      = 13
BOARD_CX     = 6   # centre row
BOARD_CY     = 6   # centre col

# ── Tuning ────────────────────────────────────────────────────────────────────
TIME_BUDGET_S             = 0.065   # tighter than v6 to leave room for net
NET_TIME_BUDGET_S         = 0.010   # max time for neural scorer
BFS_DEPTH_CAP             = 30
ESCAPE_DEPTH              = 25
BOMB_MIN_SCORE            = 1
BOMB_COMBAT_CLUSTER_SCORE = 4

# ── Neural scorer path ────────────────────────────────────────────────────────
SCORER_PATH = os.path.join(os.path.dirname(__file__), "scorer_best.pt")

# ── Feature dimension ─────────────────────────────────────────────────────────
# State context: 27 scalars (see _encode_state_context)
# Per-candidate extra: 7 scalars (see _encode_candidate_features)
STATE_DIM     = 27
CAND_DIM      = 7
TOTAL_IN_DIM  = STATE_DIM + CAND_DIM   # 34


# =============================================================================
# Neural scorer definition  (must match train_scorer.py exactly)
# =============================================================================

class ScorerNet(nn.Module if _TORCH else object):
    """
    Tiny MLP: 34 → 64 → 32 → 1.
    Scores a single (state_context, candidate_features) pair.
    ~46K parameters.
    """
    def __init__(self):
        if not _TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOTAL_IN_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32),           nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _load_scorer():
    """Load scorer weights. Returns model in eval mode, or None."""
    if not _TORCH:
        return None
    if not os.path.exists(SCORER_PATH):
        return None
    try:
        model = ScorerNet()
        model.load_state_dict(torch.load(SCORER_PATH, map_location="cpu"))
        model.eval()
        return model
    except Exception:
        return None


# =============================================================================
# Geometry helpers
# =============================================================================

def _blast_tiles(bx, by, radius, grid):
    """Tiles in this bomb's blast zone. Stops at WALL, stops+includes BOX."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        for r in range(1, radius + 1):
            tr, tc = bx + dr*r, by + dc*r
            if not (0 <= tr < h and 0 <= tc < w):
                break
            cell = grid[tr, tc]
            if cell == WALL:
                break
            tiles.add((tr, tc))
            if cell == BOX:
                break
    return tiles


def _walkable(r, c, grid, bomb_pos):
    """Can the agent step onto (r,c)?"""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if grid[r, c] in (WALL, BOX):
        return False
    if (r, c) in bomb_pos:
        return False
    return True


# =============================================================================
# Timer-aware danger map  (with chain reactions)
# =============================================================================

def _build_danger_timed(obs):
    """
    Returns (danger_by_time, danger_any).
    Chain reaction: bomb B hit by A's blast -> B.timer = min(A.timer, B.timer).
    """
    grid    = obs["map"]
    bombs_a = obs["bombs"]
    players = obs["players"]

    if len(bombs_a) == 0:
        return {}, set()

    n = len(players)
    bomb_list = []
    for b in bombs_a:
        bx, by, timer, oid = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        oid_s  = max(0, min(n-1, oid))
        radius = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
        tiles  = _blast_tiles(bx, by, radius, grid)
        bomb_list.append({'pos':(bx,by), 'timer':timer, 'tiles':tiles})

    changed = True
    while changed:
        changed = False
        for i, b1 in enumerate(bomb_list):
            for j, b2 in enumerate(bomb_list):
                if i == j: continue
                if b2['pos'] in b1['tiles'] and b1['timer'] < b2['timer']:
                    b2['timer'] = b1['timer']
                    changed = True

    danger_by_time = {}
    danger_any     = set()
    for b in bomb_list:
        t = b['timer']
        danger_by_time.setdefault(t, set())
        danger_by_time[t] |= b['tiles']
        danger_any        |= b['tiles']

    return danger_by_time, danger_any


# =============================================================================
# Timer-aware BFS (navigation)
# =============================================================================

def _bfs_timed(start, targets, grid, bomb_pos, danger_by_time,
               depth_cap=BFS_DEPTH_CAP):
    """BFS tracking arrival time. Returns first action, or None."""
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr, sc)}
    queue   = deque()
    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr+dr, sc+dc
        if not _walkable(nr, nc, grid, bomb_pos): continue
        if (nr,nc) in visited: continue
        if (nr,nc) in danger_by_time.get(1, set()): continue
        visited.add((nr,nc))
        queue.append((nr,nc,a,1))

    while queue:
        r, c, first_a, t = queue.popleft()
        if (r,c) in targets: return first_a
        if t >= depth_cap: continue
        for a in MOVE_ACTIONS:
            dr, dc = MOVES[a]
            nr, nc = r+dr, c+dc
            t_next = t+1
            if not _walkable(nr, nc, grid, bomb_pos): continue
            if (nr,nc) in visited: continue
            if (nr,nc) in danger_by_time.get(t_next, set()): continue
            visited.add((nr,nc))
            queue.append((nr,nc,first_a,t_next))

    return None


# =============================================================================
# Time-expanded escape BFS  (STOP allowed, revisit at later time slice)
# =============================================================================

def _bfs_escape(start, targets, grid, bomb_pos, danger_by_time,
                depth_cap=ESCAPE_DEPTH):
    """Returns first action toward nearest safe cell, or None."""
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr,sc,0)}
    queue   = deque([(sr,sc,None,0)])

    while queue:
        r, c, first_a, t = queue.popleft()
        if t >= depth_cap: continue
        t_next      = t + 1
        danger_next = danger_by_time.get(t_next, set())

        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r+dr, c+dc
                if not _walkable(nr, nc, grid, bomb_pos): continue
            if (nr,nc,t_next) in visited: continue
            if (nr,nc) in danger_next: continue
            fa = a if first_a is None else first_a
            if (nr,nc) in targets: return fa
            visited.add((nr,nc,t_next))
            queue.append((nr,nc,fa,t_next))

    return None


# =============================================================================
# Escape dispatcher  (F2: collect items on escape path for free)
# =============================================================================

def _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any, items=None):
    """
    3-pass escape: fully safe -> less-risky -> any move.
    F2: if items is provided, an item on the escape path is collected
        (we return toward it only when it is also a safe target).
    """
    h, w = grid.shape

    safe = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr,cc] not in (WALL,BOX)
                    and (rr,cc) not in bomb_pos
                    and (rr,cc) not in danger_any):
                safe.add((rr,cc))

    # F2: items that are also safe — pick them up on the way out
    if items:
        safe_items = items & safe
        if safe_items:
            a = _bfs_escape(pos, safe_items, grid, bomb_pos, danger_by_time,
                            depth_cap=ESCAPE_DEPTH)
            if a is not None:
                return a

    if safe:
        a = _bfs_escape(pos, safe, grid, bomb_pos, danger_by_time,
                        depth_cap=ESCAPE_DEPTH)
        if a is not None:
            return a

    urgent     = danger_by_time.get(1,set()) | danger_by_time.get(2,set())
    less_risky = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr,cc] not in (WALL,BOX)
                    and (rr,cc) not in bomb_pos
                    and (rr,cc) not in urgent):
                less_risky.add((rr,cc))

    if less_risky:
        sr, sc = pos
        vis  = {(sr,sc)}
        que  = deque()
        for a in MOVE_ACTIONS:
            dr,dc = MOVES[a]; nr,nc = sr+dr,sc+dc
            if _walkable(nr,nc,grid,bomb_pos) and (nr,nc) not in vis:
                vis.add((nr,nc)); que.append((nr,nc,a))
        while que:
            r,c,first_a = que.popleft()
            if (r,c) in less_risky: return first_a
            for a in MOVE_ACTIONS:
                dr,dc = MOVES[a]; nr,nc = r+dr,c+dc
                if _walkable(nr,nc,grid,bomb_pos) and (nr,nc) not in vis:
                    vis.add((nr,nc)); que.append((nr,nc,first_a))

    sr, sc = pos
    for a in MOVE_ACTIONS:
        dr,dc = MOVES[a]; nr,nc = sr+dr,sc+dc
        if _walkable(nr,nc,grid,bomb_pos): return a
    return STOP


# =============================================================================
# Bomb value scoring  (F1: chain-reaction bonus, F3: multi-box density)
# =============================================================================

def _bomb_value(my_r, my_c, my_radius, grid, players, agent_id, danger_any,
                bomb_pos, danger_by_time):
    """
    Score a potential bomb at (my_r, my_c).
    Returns (score, hits_box, hits_enemy).

    F1  Chain bonus: if our blast hits another live bomb position,
        those bombs' blast tiles are added to the effective blast.
        Each extra box/enemy in the chain earns full value.
    F3  Multi-box density: each box hit past the first earns +0.5 extra.
    """
    direct_blast = _blast_tiles(my_r, my_c, my_radius, grid)

    # F1: chain reactions — if blast hits a live bomb, extend coverage
    chain_blast = set(direct_blast)
    for br, bc in list(bomb_pos):
        if (br,bc) in direct_blast:
            # find that bomb's owner radius
            for i,p in enumerate(players):
                pass  # default radius 1 if no info
            chain_blast |= _blast_tiles(br, bc, 1, grid)  # conservative r=1
    # also collect chain from obs bombs array (passed via closure via grid)
    effective_blast = chain_blast

    boxes   = sum(1 for r,c in effective_blast if grid[r,c] == BOX)
    enemies = sum(
        1 for i in range(len(players))
        if i != agent_id
        and int(players[i][2]) == 1
        and (int(players[i][0]), int(players[i][1])) in effective_blast
    )

    # F3: multi-box density bonus
    box_score = boxes + max(0, boxes - 1) * 0.5

    h, w = grid.shape
    adj_safe = sum(
        1 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
        if (0 <= my_r+dr < h and 0 <= my_c+dc < w
            and grid[my_r+dr, my_c+dc] == GRASS
            and (my_r+dr, my_c+dc) not in danger_any)
    )
    corridor_pen = 2 if adj_safe <= 1 else 0
    score = box_score + enemies * 3 - corridor_pen
    return score, boxes > 0, enemies > 0


def _bomb_value_full(my_r, my_c, my_radius, grid, players, agent_id,
                     danger_any, obs_bombs, bomb_pos, danger_by_time):
    """
    Extended bomb value with full chain-radius info from obs_bombs.
    Used when we have the raw bombs array available.
    """
    direct_blast = _blast_tiles(my_r, my_c, my_radius, grid)
    n = len(players)

    # build chain: if direct blast hits another bomb, include its blast
    chain_blast = set(direct_blast)
    for b in obs_bombs:
        br, bc = int(b[0]), int(b[1])
        if (br,bc) in direct_blast:
            oid    = int(b[3])
            oid_s  = max(0, min(n-1, oid))
            r_bomb = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
            chain_blast |= _blast_tiles(br, bc, r_bomb, grid)

    boxes   = sum(1 for r,c in chain_blast if grid[r,c] == BOX)
    enemies = sum(
        1 for i in range(n)
        if i != agent_id
        and int(players[i][2]) == 1
        and (int(players[i][0]), int(players[i][1])) in chain_blast
    )
    box_score    = boxes + max(0, boxes-1) * 0.5
    chain_bonus  = 0.5 if len(chain_blast) > len(direct_blast) else 0.0

    h, w = grid.shape
    adj_safe = sum(
        1 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
        if (0 <= my_r+dr < h and 0 <= my_c+dc < w
            and grid[my_r+dr,my_c+dc] == GRASS
            and (my_r+dr,my_c+dc) not in danger_any)
    )
    corridor_pen = 2 if adj_safe <= 1 else 0
    score = box_score + enemies*3 + chain_bonus - corridor_pen
    return score, boxes > 0, enemies > 0


# =============================================================================
# Bomb escape helpers
# =============================================================================

def _nearby_bomb_count(my_r, my_c, bomb_pos, radius=3):
    """Count bombs within Manhattan radius (used for F5 eff_t scaling)."""
    return sum(
        1 for (br,bc) in bomb_pos
        if abs(br-my_r)+abs(bc-my_c) <= radius
    )


def _can_escape_after_bomb(my_r, my_c, my_radius, grid, bomb_pos,
                            danger_by_time, extra_danger_tiles=None):
    """
    Return True if agent can reach a safe cell after placing bomb here.
    F5: eff_t shrinks if many live bombs are nearby (less time to escape).
    """
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r,my_c)}

    eff_t = BOMB_TIMER - 1   # base = 6
    for t, tiles in danger_by_time.items():
        if (my_r,my_c) in tiles and t < eff_t:
            eff_t = t

    # F5: nearby bombs shorten effective escape window
    nearby = _nearby_bomb_count(my_r, my_c, bomb_pos, radius=3)
    if nearby >= 2: eff_t = max(2, eff_t - 1)
    if nearby >= 4: eff_t = max(1, eff_t - 1)

    mod = {t: set(s) for t,s in danger_by_time.items()}
    mod.setdefault(eff_t, set())
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values(): new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr,cc] not in (WALL,BOX)
                    and (rr,cc) not in new_bomb_p
                    and (rr,cc) not in new_danger_any):
                safe_cells.add((rr,cc))

    if not safe_cells: return False

    visited = {(my_r,my_c,0)}
    queue   = deque([(my_r,my_c,0)])
    while queue:
        r,c,t = queue.popleft()
        if (r,c) in safe_cells: return True
        if t >= eff_t: continue
        t_next      = t+1
        danger_next = mod.get(t_next, set())
        for a in (LEFT,RIGHT,UP,DOWN,STOP):
            if a == STOP: nr,nc = r,c
            else:
                dr,dc = MOVES[a]; nr,nc = r+dr,c+dc
                if not _walkable(nr,nc,grid,new_bomb_p): continue
            if (nr,nc,t_next) in visited: continue
            if (nr,nc) in danger_next: continue
            visited.add((nr,nc,t_next))
            queue.append((nr,nc,t_next))

    return False


def _count_escape_first_moves(my_r, my_c, my_radius, grid, bomb_pos,
                               danger_by_time, extra_danger_tiles=None):
    """C3: count distinct first-move directions that lead to safety."""
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r,my_c)}

    eff_t = BOMB_TIMER - 1
    for t,tiles in danger_by_time.items():
        if (my_r,my_c) in tiles and t < eff_t: eff_t = t

    nearby = _nearby_bomb_count(my_r, my_c, bomb_pos, radius=3)
    if nearby >= 2: eff_t = max(2, eff_t-1)

    mod = {t: set(s) for t,s in danger_by_time.items()}
    mod.setdefault(eff_t, set())
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values(): new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if (grid[rr,cc] not in (WALL,BOX)
                    and (rr,cc) not in new_bomb_p
                    and (rr,cc) not in new_danger_any):
                safe_cells.add((rr,cc))

    if not safe_cells: return 0

    count = 0
    for first_a in (LEFT,RIGHT,UP,DOWN,STOP):
        if first_a == STOP: r1,c1 = my_r,my_c
        else:
            dr,dc = MOVES[first_a]; r1,c1 = my_r+dr,my_c+dc
            if not _walkable(r1,c1,grid,new_bomb_p): continue
        if (r1,c1) in mod.get(1,set()): continue
        if (r1,c1) in safe_cells: count+=1; continue

        vis2 = {(r1,c1,1)}
        que2 = deque([(r1,c1,1)])
        found = False
        while que2 and not found:
            r,c,t = que2.popleft()
            if t >= eff_t: continue
            t_next = t+1
            dn     = mod.get(t_next,set())
            for a2 in (LEFT,RIGHT,UP,DOWN,STOP):
                if a2 == STOP: nr,nc = r,c
                else:
                    dr2,dc2 = MOVES[a2]; nr,nc = r+dr2,c+dc2
                    if not _walkable(nr,nc,grid,new_bomb_p): continue
                if (nr,nc,t_next) in vis2: continue
                if (nr,nc) in dn: continue
                if (nr,nc) in safe_cells: found=True; break
                vis2.add((nr,nc,t_next)); que2.append((nr,nc,t_next))
        if found: count+=1

    return count


# =============================================================================
# Enemy proximity helpers
# =============================================================================

def _enemy_predicted_blast(my_r, my_c, players, agent_id, grid):
    """C1: union of blast tiles for armed enemies within Manhattan 2."""
    predicted = set()
    for i,p in enumerate(players):
        if i == agent_id: continue
        if int(p[2]) != 1: continue
        if int(p[3]) <= 0: continue
        er,ec = int(p[0]),int(p[1])
        if abs(er-my_r)+abs(ec-my_c) > 2: continue
        radius = max(1,min(MAX_RADIUS, 1+int(p[4])))
        predicted |= _blast_tiles(er,ec,radius,grid)
    return predicted


def _count_nearby_armed(my_r, my_c, players, agent_id):
    """Count armed enemies within Manhattan 2."""
    count = 0
    for i,p in enumerate(players):
        if i == agent_id: continue
        if int(p[2]) != 1: continue
        if int(p[3]) <= 0: continue
        er,ec = int(p[0]),int(p[1])
        if abs(er-my_r)+abs(ec-my_c) <= 2: count+=1
    return count


def _enemy_adjacent(my_r, my_c, players, agent_id):
    """True if any live armed enemy is within Manhattan 2."""
    return _count_nearby_armed(my_r, my_c, players, agent_id) > 0


# =============================================================================
# Reachable safe count  (anti-corner)
# =============================================================================

def _reachable_safe_count(pos, grid, bomb_pos, danger_any, depth=4):
    """BFS up to depth steps; count reachable cells not in danger_any."""
    visited = {pos}
    queue   = deque([(pos,0)])
    count   = 0
    while queue:
        (r,c),d = queue.popleft()
        if (r,c) not in danger_any: count+=1
        if d >= depth: continue
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc = r+dr,c+dc
            if (nr,nc) not in visited and _walkable(nr,nc,grid,bomb_pos):
                visited.add((nr,nc)); queue.append(((nr,nc),d+1))
    return count


# =============================================================================
# Safe fallback  (F7: map-control term, anti-repeat)
# =============================================================================

def _safe_fallback(pos, grid, bomb_pos, danger_by_time, step_count, agent_id,
                   last_pos=None):
    """
    Primary: reachable_safe_count*10 + open_neighbours.
    F7: +centre_bonus = max(0, 6 - manhattan_to_centre) * 0.3 (late-game pull).
    C4c: -15 for returning to last_pos.
    """
    sr, sc = pos
    imm_danger = danger_by_time.get(1, set())
    any_danger = set()
    for tiles in danger_by_time.values(): any_danger |= tiles

    def _score(r, c, is_stop, a):
        reach  = _reachable_safe_count((r,c), grid, bomb_pos, any_danger, depth=4)
        open_n = sum(
            1 for dr2,dc2 in ((-1,0),(1,0),(0,-1),(0,1))
            if _walkable(r+dr2,c+dc2,grid,bomb_pos)
        )
        pen        = 5 if (r,c) in any_danger else 0
        rep_pen    = 15 if (not is_stop) and last_pos is not None \
                        and (r,c) == last_pos else 0
        # F7: gentle centre pull
        centre_b   = max(0, 6 - (abs(r-BOARD_CX)+abs(c-BOARD_CY))) * 0.3
        tb = (step_count*13 + agent_id*7 + r*5 + c*3 + a*11) % 97
        return reach*10 + open_n - pen - rep_pen + centre_b, open_n, tb

    candidates        = []
    dead_end_cands    = []

    if (sr,sc) not in imm_danger:
        sc_score,sc_open,sc_tb = _score(sr,sc,True,STOP)
        candidates.append((STOP,sc_score,True,sc_open,sc_tb))

    for a in MOVE_ACTIONS:
        dr,dc = MOVES[a]; nr,nc = sr+dr,sc+dc
        if not _walkable(nr,nc,grid,bomb_pos): continue
        if (nr,nc) in imm_danger: continue
        mv_score,mv_open,mv_tb = _score(nr,nc,False,a)
        if mv_open <= 1:
            dead_end_cands.append((a,mv_score,False,mv_open,mv_tb))
        else:
            candidates.append((a,mv_score,False,mv_open,mv_tb))

    if not candidates: candidates = dead_end_cands
    if not candidates: return STOP

    best_score = max(c[1] for c in candidates)
    best       = [c for c in candidates if c[1] == best_score]
    non_stop   = [c for c in best if not c[2]]
    pool       = non_stop if non_stop else best
    pool.sort(key=lambda x: x[4])
    return pool[0][0]


# =============================================================================
# Box targeting  (F6: prefer spots adjacent to multiple boxes)
# =============================================================================

def _scored_box_spots(grid, bomb_pos, danger_any):
    """
    Returns list of (score, (r,c)) for all box-adjacent safe spots.
    F6: score = number of distinct boxes reachable via bomb from that spot.
    Higher = more chain value.
    """
    h, w = grid.shape
    spot_scores = {}
    for r in range(h):
        for c in range(w):
            if grid[r,c] != BOX: continue
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr,nc = r+dr,c+dc
                if not (0<=nr<h and 0<=nc<w): continue
                if grid[nr,nc] in (WALL,BOX): continue
                if (nr,nc) in bomb_pos: continue
                if (nr,nc) in danger_any: continue
                # count boxes a default-radius bomb would hit from here
                box_count = sum(
                    1 for tr,tc in _blast_tiles(nr,nc,1,grid)
                    if grid[tr,tc] == BOX
                )
                prev = spot_scores.get((nr,nc), 0)
                spot_scores[(nr,nc)] = max(prev, box_count)

    return spot_scores  # dict (r,c)->score


# =============================================================================
# Feature encoding for neural scorer
# =============================================================================

def _encode_state_context(my_r, my_c, my_radius, bombs_left,
                          grid, players, agent_id,
                          danger_any, bomb_pos,
                          danger_by_time, step):
    """
    27 floats describing the current state from agent's perspective.
    Must be identical in agent.py and train_scorer.py.
    """
    h, w = grid.shape
    n_players = len(players)

    # 1. my position normalised
    r_n  = my_r / (h-1)
    c_n  = my_c / (w-1)

    # 2. distance to centre
    dist_centre = (abs(my_r-BOARD_CX)+abs(my_c-BOARD_CY)) / 12.0

    # 3. danger flags
    in_imm = float((my_r,my_c) in danger_by_time.get(1,set()))
    in_any = float((my_r,my_c) in danger_any)

    # 4. bombs_left, radius (normalised)
    bl_n  = min(bombs_left, MAX_CAPACITY) / MAX_CAPACITY
    rad_n = my_radius / MAX_RADIUS

    # 5. live enemies count
    n_alive = sum(
        1 for i,p in enumerate(players)
        if i != agent_id and int(p[2]) == 1
    ) / 3.0

    # 6. nearby armed enemies (Manhattan<=3)
    armed_near = sum(
        1 for i,p in enumerate(players)
        if i != agent_id and int(p[2])==1 and int(p[3])>0
        and abs(int(p[0])-my_r)+abs(int(p[1])-my_c)<=3
    ) / 3.0

    # 7. nearest enemy distance (BFS)
    enemies = {
        (int(players[i][0]),int(players[i][1]))
        for i in range(n_players)
        if i != agent_id and int(players[i][2])==1
    }
    def _bfs_dist(start, targets, depth=16):
        if not targets: return 1.0
        vis = {start}; q = deque([(start,0)])
        while q:
            pos,d = q.popleft()
            if pos in targets: return d/16.0
            if d>=depth: continue
            for dr2,dc2 in ((-1,0),(1,0),(0,-1),(0,1)):
                nr2,nc2 = pos[0]+dr2,pos[1]+dc2
                if (nr2,nc2) not in vis and _walkable(nr2,nc2,grid,bomb_pos):
                    vis.add((nr2,nc2)); q.append(((nr2,nc2),d+1))
        return 1.0
    enemy_dist = _bfs_dist((my_r,my_c), enemies)

    # 8. nearest item distance
    items = {
        (r,c) for r in range(h) for c in range(w)
        if grid[r,c] in (ITEM_RADIUS,ITEM_CAPACITY)
    }
    item_dist = _bfs_dist((my_r,my_c), items)

    # 9. reachable safe count (normalised)
    reach = _reachable_safe_count((my_r,my_c), grid, bomb_pos, danger_any, depth=4)
    reach_n = min(reach, 40) / 40.0

    # 10. live bomb count on map
    n_bombs_n = min(len(bomb_pos), 10) / 10.0

    # 11. boxes remaining (normalised)
    n_boxes = sum(1 for r in range(h) for c in range(w) if grid[r,c]==BOX)
    boxes_n = n_boxes / 60.0

    # 12. step ratio
    step_n = min(step, 500) / 500.0

    # 13. box adjacency: how many boxes are adjacent to me right now
    adj_boxes = sum(
        1 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
        if 0<=my_r+dr<h and 0<=my_c+dc<w and grid[my_r+dr,my_c+dc]==BOX
    ) / 4.0

    # 14-17. nearest 4 enemy positions relative to me (sorted by proximity)
    enemy_list = sorted(
        [(int(p[0]),int(p[1])) for i,p in enumerate(players)
         if i!=agent_id and int(p[2])==1],
        key=lambda ep: abs(ep[0]-my_r)+abs(ep[1]-my_c)
    )
    rel_enemies = []
    for ep in enemy_list[:2]:
        rel_enemies.append((ep[0]-my_r)/12.0)
        rel_enemies.append((ep[1]-my_c)/12.0)
    while len(rel_enemies) < 4: rel_enemies.append(0.0)

    # 18. is_stuck hint (1 if oscillating)
    # (filled in by caller if known, default 0)
    is_stuck = 0.0

    # 19. combat_mode hint
    combat = float(armed_near > 0)

    vec = [
        r_n, c_n, dist_centre,
        in_imm, in_any,
        bl_n, rad_n,
        n_alive, armed_near,
        enemy_dist, item_dist,
        reach_n, n_bombs_n, boxes_n, step_n,
        adj_boxes,
        rel_enemies[0], rel_enemies[1], rel_enemies[2], rel_enemies[3],
        is_stuck, combat,
        # 5 spare zeros (pad to STATE_DIM=27)
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert len(vec) == STATE_DIM, f"STATE_DIM mismatch: {len(vec)}"
    return vec


def _encode_candidate_features(action, my_r, my_c, my_radius,
                                grid, players, agent_id,
                                danger_any, bomb_pos, danger_by_time,
                                obs_bombs):
    """
    7 floats describing what happens if this candidate action is taken.
    Must be identical in agent.py and train_scorer.py.
    """
    h, w = grid.shape

    # 1. action id normalised
    act_n = action / 5.0

    # 2. destination cell features
    if action == STOP:
        nr, nc = my_r, my_c
    elif action == PLACE_BOMB:
        nr, nc = my_r, my_c
    else:
        dr, dc = MOVES[action]
        nr, nc = my_r+dr, my_c+dc

    # 3. destination reachability (0 if not walkable/bomb)
    dest_ok = float(_walkable(nr,nc,grid,bomb_pos)) if action not in (STOP,PLACE_BOMB) else 1.0

    # 4. destination danger
    dest_danger = float((nr,nc) in danger_any)

    # 5. if PLACE_BOMB: bomb value score normalised
    if action == PLACE_BOMB:
        bval, _, _ = _bomb_value_full(
            my_r,my_c,my_radius,grid,players,agent_id,
            danger_any,obs_bombs,bomb_pos,danger_by_time
        )
        bval_n = min(max(bval,0),15) / 15.0
    else:
        bval_n = 0.0

    # 6. destination open neighbours
    if action not in (STOP, PLACE_BOMB):
        open_n = sum(
            1 for dr2,dc2 in ((-1,0),(1,0),(0,-1),(0,1))
            if _walkable(nr+dr2,nc+dc2,grid,bomb_pos)
        ) / 4.0
    else:
        open_n = sum(
            1 for dr2,dc2 in ((-1,0),(1,0),(0,-1),(0,1))
            if _walkable(my_r+dr2,my_c+dc2,grid,bomb_pos)
        ) / 4.0

    # 7. destination is item
    dest_item = float(
        0 <= nr < h and 0 <= nc < w
        and grid[nr,nc] in (ITEM_RADIUS,ITEM_CAPACITY)
    )

    return [act_n, dest_ok, dest_danger, bval_n, open_n, dest_item, 0.0]


# =============================================================================
# Neural candidate scorer
# =============================================================================

def _score_candidates_with_net(net, state_ctx, candidates,
                                my_r, my_c, my_radius,
                                grid, players, agent_id,
                                danger_any, bomb_pos, danger_by_time,
                                obs_bombs):
    """
    Score each candidate action with the neural net.
    Returns action with highest score, or None on failure.
    """
    if not _TORCH or net is None or not candidates:
        return None
    try:
        state_t = torch.tensor(state_ctx, dtype=torch.float32)
        scores  = []
        for a in candidates:
            cand_f = _encode_candidate_features(
                a, my_r, my_c, my_radius,
                grid, players, agent_id,
                danger_any, bomb_pos, danger_by_time,
                obs_bombs
            )
            inp = torch.cat([state_t, torch.tensor(cand_f, dtype=torch.float32)])
            with torch.no_grad():
                s = float(net(inp.unsqueeze(0)).item())
            scores.append(s)
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]
    except Exception:
        return None


# =============================================================================
# Candidate generation  (rule layer output)
# =============================================================================

def _generate_candidates(pos, my_r, my_c, my_radius, bombs_left,
                          grid, players, agent_id,
                          danger_any, bomb_pos, danger_by_time,
                          extra_danger, combat_mode, nearby_armed,
                          obs_bombs, t0):
    """
    Return list of candidate actions that pass hard safety gates.
    This is the rule layer's gift to the neural scorer.

    Candidates are ordered by rule priority so argmax over neural scores
    beats the rule baseline when the net has learned something useful.
    """
    cands = []
    h, w  = grid.shape

    # ── Movement candidates ────────────────────────────────────────────────
    for a in MOVE_ACTIONS:
        dr,dc = MOVES[a]; nr,nc = my_r+dr,my_c+dc
        if not _walkable(nr,nc,grid,bomb_pos): continue
        if (nr,nc) in danger_by_time.get(1,set()): continue  # immediate death
        cands.append(a)

    # ── STOP: only candidate if not in immediate danger ────────────────────
    if pos not in danger_by_time.get(1,set()):
        cands.append(STOP)

    # ── PLACE_BOMB: only if passes all safety gates ────────────────────────
    if (bombs_left > 0
            and pos not in bomb_pos
            and time.perf_counter()-t0 < TIME_BUDGET_S):
        bval,hits_box,hits_enemy = _bomb_value_full(
            my_r,my_c,my_radius,grid,players,agent_id,
            danger_any,obs_bombs,bomb_pos,danger_by_time
        )
        threshold = BOMB_COMBAT_CLUSTER_SCORE if nearby_armed>=2 else BOMB_MIN_SCORE
        if bval >= threshold and (hits_box or hits_enemy):
            can_esc = _can_escape_after_bomb(
                my_r,my_c,my_radius,grid,bomb_pos,danger_by_time,
                extra_danger_tiles=extra_danger
            )
            if can_esc and combat_mode:
                can_esc = _count_escape_first_moves(
                    my_r,my_c,my_radius,grid,bomb_pos,danger_by_time,
                    extra_danger_tiles=extra_danger
                ) >= 2
            if can_esc:
                cands.append(PLACE_BOMB)

    return cands


# =============================================================================
# Agent
# =============================================================================

class Agent:
    team_id = "ClaudeAgent"

    def __init__(self, agent_id: int, verbose: bool = False):
        self.agent_id    = int(agent_id)
        self.step_count  = 0
        self.pos_history = deque(maxlen=20)
        self.last_pos    = None
        self._scorer     = _load_scorer()
        if self._scorer is not None and verbose:
            print(f"[Agent {agent_id}] Neural scorer loaded from {SCORER_PATH}")
        elif verbose is True:
            print(f"[Agent {agent_id}] No scorer found — pure rule agent")

    def act(self, obs: dict) -> int:
        try:
            return int(self._act_impl(obs))
        except Exception:
            return STOP
        finally:
            self.step_count += 1

    def _act_impl(self, obs: dict) -> int:
        t0 = time.perf_counter()

        grid    = obs["map"]
        players = obs["players"]
        bombs_a = obs["bombs"]

        me           = players[self.agent_id]
        my_r,my_c    = int(me[0]),int(me[1])
        alive        = int(me[2])
        bombs_left   = int(me[3])
        radius_bonus = int(me[4])

        if not alive: return STOP

        my_radius = max(1, min(MAX_RADIUS, 1+radius_bonus))
        pos       = (my_r, my_c)

        self.pos_history.append(pos)
        prev_pos      = self.last_pos
        self.last_pos = pos

        danger_by_time, danger_any = _build_danger_timed(obs)
        bomb_pos    = {(int(b[0]),int(b[1])) for b in bombs_a}
        nearby_armed = _count_nearby_armed(my_r,my_c,players,self.agent_id)
        combat_mode  = nearby_armed >= 1

        extra_danger = set()
        if combat_mode:
            extra_danger = _enemy_predicted_blast(
                my_r,my_c,players,self.agent_id,grid)

        is_stuck = (
            len(self.pos_history) >= 16
            and len(set(self.pos_history)) <= 3
            and len(bombs_a) == 0
            and pos not in danger_any
        )

        h, w = grid.shape
        items = {(r,c) for r in range(h) for c in range(w)
                 if grid[r,c] in (ITEM_RADIUS,ITEM_CAPACITY)}

        # ── PRIORITY 1: ESCAPE (hard override, no net) ────────────────────
        in_immediate = pos in danger_by_time.get(1,set())
        in_any       = pos in danger_any
        if in_immediate or in_any:
            # F2: pass items so we can collect on the escape path
            return _escape_timed(pos,grid,bomb_pos,danger_by_time,danger_any,
                                  items=items)

        # ── PRIORITY 2: PICK UP ITEM (on safe path only) ──────────────────
        if items and time.perf_counter()-t0 < TIME_BUDGET_S:
            a = _bfs_timed(pos,items,grid,bomb_pos,danger_by_time)
            if a is not None and a != STOP:
                return a

        # ── PRIORITY 3: UNSTUCK BOMB (farm_mode only) ─────────────────────
        if is_stuck and not combat_mode and bombs_left>0 and pos not in bomb_pos:
            if time.perf_counter()-t0 < TIME_BUDGET_S:
                blast   = _blast_tiles(my_r,my_c,my_radius,grid)
                has_box = any(grid[r][c]==BOX for r,c in blast)
                if has_box and _can_escape_after_bomb(
                        my_r,my_c,my_radius,grid,bomb_pos,
                        danger_by_time,extra_danger_tiles=None):
                    return PLACE_BOMB

        # ── NEURAL SCORING ZONE ───────────────────────────────────────────
        # Generate rule-safe candidates; let the net pick the best.
        # Falls back to deterministic rule priority if net is absent/slow.
        if time.perf_counter()-t0 < TIME_BUDGET_S:
            cands = _generate_candidates(
                pos,my_r,my_c,my_radius,bombs_left,
                grid,players,self.agent_id,
                danger_any,bomb_pos,danger_by_time,
                extra_danger,combat_mode,nearby_armed,
                bombs_a,t0
            )

            if self._scorer is not None and cands:
                t_net = time.perf_counter()
                if t_net - t0 < TIME_BUDGET_S - NET_TIME_BUDGET_S:
                    state_ctx = _encode_state_context(
                        my_r,my_c,my_radius,bombs_left,
                        grid,players,self.agent_id,
                        danger_any,bomb_pos,danger_by_time,
                        self.step_count
                    )
                    state_ctx[20] = float(is_stuck)  # fill is_stuck slot
                    state_ctx[21] = float(combat_mode)

                    net_action = _score_candidates_with_net(
                        self._scorer, state_ctx, cands,
                        my_r,my_c,my_radius,
                        grid,players,self.agent_id,
                        danger_any,bomb_pos,danger_by_time,
                        bombs_a
                    )
                    if net_action is not None:
                        return net_action

        # ── RULE FALLBACK (no net or net skipped) ─────────────────────────

        # Priority 3: place bomb if scored
        if bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter()-t0 < TIME_BUDGET_S:
                bval,hits_box,hits_enemy = _bomb_value_full(
                    my_r,my_c,my_radius,grid,players,self.agent_id,
                    danger_any,bombs_a,bomb_pos,danger_by_time
                )
                threshold = BOMB_COMBAT_CLUSTER_SCORE if nearby_armed>=2 else BOMB_MIN_SCORE
                if bval>=threshold and (hits_box or hits_enemy):
                    can_esc = _can_escape_after_bomb(
                        my_r,my_c,my_radius,grid,bomb_pos,
                        danger_by_time,extra_danger_tiles=extra_danger)
                    if can_esc and combat_mode:
                        can_esc = _count_escape_first_moves(
                            my_r,my_c,my_radius,grid,bomb_pos,
                            danger_by_time,extra_danger_tiles=extra_danger)>=2
                    if can_esc: return PLACE_BOMB

        # Priority 4: farm boxes — F6 prefer high-cluster spots
        if time.perf_counter()-t0 < TIME_BUDGET_S:
            spot_scores = _scored_box_spots(grid,bomb_pos,danger_any)
            if spot_scores and pos not in spot_scores:
                # try best cluster spots first, fall back to any box spot
                best_val  = max(spot_scores.values())
                top_spots = {s for s,v in spot_scores.items() if v==best_val}
                a = _bfs_timed(pos,top_spots,grid,bomb_pos,danger_by_time)
                if a is None or a == STOP:
                    a = _bfs_timed(pos,set(spot_scores.keys()),
                                   grid,bomb_pos,danger_by_time)
                if a is not None and a != STOP:
                    return a

        # Priority 5: chase nearest enemy (F4: explicit time check)
        if time.perf_counter()-t0 < TIME_BUDGET_S:
            if not _enemy_adjacent(my_r,my_c,players,self.agent_id):
                enemy_tiles = {
                    (int(players[i][0]),int(players[i][1]))
                    for i in range(len(players))
                    if i!=self.agent_id and int(players[i][2])==1
                }
                if enemy_tiles:
                    a = _bfs_timed(pos,enemy_tiles,grid,bomb_pos,danger_by_time)
                    if a is not None and a != STOP:
                        return a

        return _safe_fallback(pos,grid,bomb_pos,danger_by_time,
                              self.step_count,self.agent_id,last_pos=prev_pos)