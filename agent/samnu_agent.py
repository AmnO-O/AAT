"""
agent.py -- Bomberland Rule-Based Agent v6 "Context-Aware Bombing"
Base: v5 (single-box bombing, BOMB_MIN_SCORE=1).
Changes over v5:
  C1. Enemy-bomb prediction in _can_escape_after_bomb: armed enemies (Manhattan<=2,
      bombs_left>0) are assumed to place a bomb simultaneously; their predicted
      blast is added to the danger map before BFS escape. Prevents trap deaths.
  C2. Combat-cluster gate: bomb threshold rises to BOMB_COMBAT_CLUSTER_SCORE=4
      when >=2 armed enemies are within Manhattan 2. Stops escalation into cluster.
  C3. Escape redundancy (combat_mode only): require >=2 distinct first-move
      directions that each lead to safety before placing a bomb.
  C4. Stuck-detector + unstuck-bomb + anti-repeat:
      - pos_history deque(maxlen=20): detect oscillation (<=3 unique positions
        in last 16 steps, no live bombs, not in danger).
      - Unstuck-bomb: if stuck AND farm_mode AND box in blast AND can escape ->
        place bomb regardless of corridor_pen / threshold.
      - Anti-repeat: _safe_fallback penalises returning to previous cell (-15).
  Conflict guarantee: C4b (unstuck-bomb) fires ONLY in farm_mode;
  C2/C3 fire ONLY in combat_mode -> mutually exclusive, no interference.
"""
import time
from collections import deque

# --- Action constants ---------------------------------------------------------
STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB = 0, 1, 2, 3, 4, 5

MOVES = {
    STOP:  ( 0,  0),
    LEFT:  (-1,  0),
    RIGHT: ( 1,  0),
    UP:    ( 0, -1),
    DOWN:  ( 0,  1),
}
MOVE_ACTIONS = [LEFT, RIGHT, UP, DOWN]

# --- Map cell types -----------------------------------------------------------
GRASS, WALL, BOX, ITEM_RADIUS, ITEM_CAPACITY = 0, 1, 2, 3, 4

# --- Game constants -----------------------------------------------------------
BOMB_TIMER   = 7
MAX_RADIUS   = 5
MAX_CAPACITY = 5

# --- Tuning ------------------------------------------------------------------
TIME_BUDGET_S  = 0.070
BFS_DEPTH_CAP  = 30
ESCAPE_DEPTH   = 25
BOMB_MIN_SCORE = 1
# C2: elevated threshold when >=2 armed enemies within Manhattan 2
BOMB_COMBAT_CLUSTER_SCORE = 4


# =============================================================================
# Geometry helpers
# =============================================================================

def _blast_tiles(bx, by, radius, grid):
    """Tiles in this bomb's blast zone. Stops at WALL, stops+includes BOX."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            tr, tc = bx + dr * r, by + dc * r
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
    """Can the agent step onto (r,c)? Blocked by WALL, BOX, and live bomb cells."""
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if grid[r, c] in (WALL, BOX):
        return False
    if (r, c) in bomb_pos:
        return False
    return True


# =============================================================================
# Timer-aware danger map
# =============================================================================

def _build_danger_timed(obs):
    """
    Returns (danger_by_time, danger_any).
    danger_by_time[t] = set of tiles whose bomb(s) explode at step t from now.
    Chain reaction: bomb B hit by blast of A -> B.effective_timer = min(A,B).
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
        oid_s  = max(0, min(n - 1, oid))
        radius = max(1, min(MAX_RADIUS, 1 + int(players[oid_s][4])))
        tiles  = _blast_tiles(bx, by, radius, grid)
        bomb_list.append({'pos': (bx, by), 'timer': timer, 'tiles': tiles})

    changed = True
    while changed:
        changed = False
        for i, b1 in enumerate(bomb_list):
            for j, b2 in enumerate(bomb_list):
                if i == j:
                    continue
                if b2['pos'] in b1['tiles'] and b1['timer'] < b2['timer']:
                    b2['timer'] = b1['timer']
                    changed = True

    danger_by_time = {}
    danger_any     = set()
    for b in bomb_list:
        t = b['timer']
        if t not in danger_by_time:
            danger_by_time[t] = set()
        danger_by_time[t] |= b['tiles']
        danger_any        |= b['tiles']

    return danger_by_time, danger_any


# =============================================================================
# Timer-aware BFS
# =============================================================================

def _bfs_timed(start, targets, grid, bomb_pos, danger_by_time, depth_cap=BFS_DEPTH_CAP):
    """
    BFS tracking arrival time. Valid step iff cell not in danger_by_time[arrival_t].
    Returns first action toward nearest target, or None.
    """
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr, sc)}
    queue   = deque()

    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if not _walkable(nr, nc, grid, bomb_pos):
            continue
        if (nr, nc) in visited:
            continue
        if (nr, nc) in danger_by_time.get(1, set()):
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, a, 1))

    while queue:
        r, c, first_a, t = queue.popleft()
        if (r, c) in targets:
            return first_a
        if t >= depth_cap:
            continue
        for a in MOVE_ACTIONS:
            dr, dc = MOVES[a]
            nr, nc = r + dr, c + dc
            t_next = t + 1
            if not _walkable(nr, nc, grid, bomb_pos):
                continue
            if (nr, nc) in visited:
                continue
            if (nr, nc) in danger_by_time.get(t_next, set()):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, first_a, t_next))

    return None


def _bfs_escape(start, targets, grid, bomb_pos, danger_by_time, depth_cap=ESCAPE_DEPTH):
    """
    Time-expanded escape search (from v4).
    visited keyed by (row, col, time): a tile may be revisited at a later,
    safer time slice. WAIT (STOP) allowed.
    Returns first action toward nearest reachable target, or None.
    """
    sr, sc = start
    if start in targets:
        return STOP

    visited = {(sr, sc, 0)}
    queue   = deque([(sr, sc, None, 0)])

    while queue:
        r, c, first_a, t = queue.popleft()
        if t >= depth_cap:
            continue
        t_next = t + 1
        danger_next = danger_by_time.get(t_next, set())

        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if not _walkable(nr, nc, grid, bomb_pos):
                    continue
            if (nr, nc, t_next) in visited:
                continue
            if (nr, nc) in danger_next:
                continue
            fa = a if first_a is None else first_a
            if (nr, nc) in targets:
                return fa
            visited.add((nr, nc, t_next))
            queue.append((nr, nc, fa, t_next))

    return None


# =============================================================================
# Escape
# =============================================================================

def _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any):
    """3-pass escape: fully safe -> urgency-only -> any move."""
    h, w = grid.shape

    safe = set()
    for rr in range(h):
        for cc in range(w):
            if grid[rr, cc] not in (WALL, BOX) and (rr, cc) not in bomb_pos \
                    and (rr, cc) not in danger_any:
                safe.add((rr, cc))

    if safe:
        a = _bfs_escape(pos, safe, grid, bomb_pos, danger_by_time, depth_cap=ESCAPE_DEPTH)
        if a is not None:
            return a

    urgent = danger_by_time.get(1, set()) | danger_by_time.get(2, set())
    less_risky = set()
    for rr in range(h):
        for cc in range(w):
            if grid[rr, cc] not in (WALL, BOX) and (rr, cc) not in bomb_pos \
                    and (rr, cc) not in urgent:
                less_risky.add((rr, cc))

    if less_risky:
        sr, sc = pos
        visited = {(sr, sc)}
        queue   = deque()
        for a in MOVE_ACTIONS:
            dr, dc = MOVES[a]
            nr, nc = sr + dr, sc + dc
            if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, a))
        while queue:
            r, c, first_a = queue.popleft()
            if (r, c) in less_risky:
                return first_a
            for a in MOVE_ACTIONS:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if _walkable(nr, nc, grid, bomb_pos) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, first_a))

    sr, sc = pos
    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if _walkable(nr, nc, grid, bomb_pos):
            return a
    return STOP


# =============================================================================
# Bomb helpers
# =============================================================================

def _enemy_predicted_blast(my_r, my_c, players, agent_id, grid):
    """
    C1: Return union of predicted blast tiles for every armed enemy within
    Manhattan 2 of (my_r, my_c). Simulates simultaneous enemy bombs.
    """
    predicted = set()
    for i, p in enumerate(players):
        if i == agent_id:
            continue
        if int(p[2]) != 1:
            continue
        if int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) > 2:
            continue
        radius = max(1, min(MAX_RADIUS, 1 + int(p[4])))
        predicted |= _blast_tiles(er, ec, radius, grid)
    return predicted


def _can_escape_after_bomb(my_r, my_c, my_radius, grid, bomb_pos, danger_by_time,
                            extra_danger_tiles=None):
    """
    Return True if agent can reach a safe cell within eff_t steps after placing bomb.
    C1: extra_danger_tiles (predicted enemy blasts) merged into danger at eff_t.
    """
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r, my_c)}

    eff_t = BOMB_TIMER - 1  # 6
    for t, tiles in danger_by_time.items():
        if (my_r, my_c) in tiles and t < eff_t:
            eff_t = t

    mod = {t: set(s) for t, s in danger_by_time.items()}
    if eff_t not in mod:
        mod[eff_t] = set()
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values():
        new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if grid[rr, cc] not in (WALL, BOX) and (rr, cc) not in new_bomb_p \
                    and (rr, cc) not in new_danger_any:
                safe_cells.add((rr, cc))

    if not safe_cells:
        return False

    visited = {(my_r, my_c, 0)}
    queue   = deque([(my_r, my_c, 0)])

    while queue:
        r, c, t = queue.popleft()
        if (r, c) in safe_cells:
            return True
        if t >= eff_t:
            continue
        t_next = t + 1
        danger_next = mod.get(t_next, set())
        for a in (LEFT, RIGHT, UP, DOWN, STOP):
            if a == STOP:
                nr, nc = r, c
            else:
                dr, dc = MOVES[a]
                nr, nc = r + dr, c + dc
                if not _walkable(nr, nc, grid, new_bomb_p):
                    continue
            if (nr, nc, t_next) in visited:
                continue
            if (nr, nc) in danger_next:
                continue
            visited.add((nr, nc, t_next))
            queue.append((nr, nc, t_next))

    return False


def _count_escape_first_moves(my_r, my_c, my_radius, grid, bomb_pos, danger_by_time,
                               extra_danger_tiles=None):
    """
    C3: Count distinct first-move directions that lead to a safe cell after
    placing a bomb. Uses same modified danger map as _can_escape_after_bomb.
    Returns int 0..5.
    """
    new_blast  = _blast_tiles(my_r, my_c, my_radius, grid)
    new_bomb_p = set(bomb_pos) | {(my_r, my_c)}

    eff_t = BOMB_TIMER - 1
    for t, tiles in danger_by_time.items():
        if (my_r, my_c) in tiles and t < eff_t:
            eff_t = t

    mod = {t: set(s) for t, s in danger_by_time.items()}
    if eff_t not in mod:
        mod[eff_t] = set()
    mod[eff_t] |= new_blast
    if extra_danger_tiles:
        mod[eff_t] |= extra_danger_tiles

    new_danger_any = set()
    for tiles in mod.values():
        new_danger_any |= tiles

    h, w = grid.shape
    safe_cells = set()
    for rr in range(h):
        for cc in range(w):
            if grid[rr, cc] not in (WALL, BOX) and (rr, cc) not in new_bomb_p \
                    and (rr, cc) not in new_danger_any:
                safe_cells.add((rr, cc))

    if not safe_cells:
        return 0

    count = 0
    for first_a in (LEFT, RIGHT, UP, DOWN, STOP):
        if first_a == STOP:
            r1, c1 = my_r, my_c
        else:
            dr, dc = MOVES[first_a]
            r1, c1 = my_r + dr, my_c + dc
            if not _walkable(r1, c1, grid, new_bomb_p):
                continue

        if (r1, c1) in mod.get(1, set()):
            continue

        if (r1, c1) in safe_cells:
            count += 1
            continue

        visited2 = {(r1, c1, 1)}
        queue2   = deque([(r1, c1, 1)])
        found    = False
        while queue2 and not found:
            r, c, t = queue2.popleft()
            if t >= eff_t:
                continue
            t_next = t + 1
            danger_next = mod.get(t_next, set())
            for a2 in (LEFT, RIGHT, UP, DOWN, STOP):
                if a2 == STOP:
                    nr, nc = r, c
                else:
                    dr2, dc2 = MOVES[a2]
                    nr, nc = r + dr2, c + dc2
                    if not _walkable(nr, nc, grid, new_bomb_p):
                        continue
                if (nr, nc, t_next) in visited2:
                    continue
                if (nr, nc) in danger_next:
                    continue
                if (nr, nc) in safe_cells:
                    found = True
                    break
                visited2.add((nr, nc, t_next))
                queue2.append((nr, nc, t_next))

        if found:
            count += 1

    return count


# =============================================================================
# Reachable safe count (anti-corner)
# =============================================================================

def _reachable_safe_count(pos, grid, bomb_pos, danger_any, depth=4):
    """
    BFS up to depth steps from pos. Count cells not in danger_any.
    Used to penalise moves leading into corner/corridor traps.
    """
    visited = {pos}
    queue   = deque([(pos, 0)])
    count   = 0
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) not in danger_any:
            count += 1
        if d >= depth:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and _walkable(nr, nc, grid, bomb_pos):
                visited.add((nr, nc))
                queue.append(((nr, nc), d + 1))
    return count


# =============================================================================
# Safe fallback (anti-corner + C4c anti-repeat)
# =============================================================================

def _safe_fallback(pos, grid, bomb_pos, danger_by_time, step_count, agent_id,
                   last_pos=None):
    """
    Primary score = reachable_safe_count(depth=4) * 10 + open_neighbors.
    Hard-skip dead-end moves (open_neighbors <= 1) unless all moves are dead-ends.
    Prefer non-STOP; deterministic tiebreak.
    C4c: penalise returning to last_pos by -15 (anti-oscillation).
    """
    sr, sc = pos
    imm_danger = danger_by_time.get(1, set())
    any_danger = set()
    for tiles in danger_by_time.values():
        any_danger |= tiles

    def _score(r, c, is_stop, a):
        reach  = _reachable_safe_count((r, c), grid, bomb_pos, any_danger, depth=4)
        open_n = sum(
            1 for dr2, dc2 in ((-1,0),(1,0),(0,-1),(0,1))
            if _walkable(r+dr2, c+dc2, grid, bomb_pos)
        )
        penalty    = 5 if (r, c) in any_danger else 0
        repeat_pen = 15 if (not is_stop) and last_pos is not None \
                          and (r, c) == last_pos else 0
        tb = (step_count * 13 + agent_id * 7 + r * 5 + c * 3 + a * 11) % 97
        return reach * 10 + open_n - penalty - repeat_pen, open_n, tb

    candidates = []

    if (sr, sc) not in imm_danger:
        sc_score, sc_open, sc_tb = _score(sr, sc, True, STOP)
        candidates.append((STOP, sc_score, True, sc_open, sc_tb))

    dead_end_candidates = []
    for a in MOVE_ACTIONS:
        dr, dc = MOVES[a]
        nr, nc = sr + dr, sc + dc
        if not _walkable(nr, nc, grid, bomb_pos):
            continue
        if (nr, nc) in imm_danger:
            continue
        mv_score, mv_open, mv_tb = _score(nr, nc, False, a)
        if mv_open <= 1:
            dead_end_candidates.append((a, mv_score, False, mv_open, mv_tb))
        else:
            candidates.append((a, mv_score, False, mv_open, mv_tb))

    if not candidates:
        candidates = dead_end_candidates

    if not candidates:
        return STOP

    best_score = max(c[1] for c in candidates)
    best = [c for c in candidates if c[1] == best_score]
    non_stop = [c for c in best if not c[2]]
    pool = non_stop if non_stop else best
    pool.sort(key=lambda x: x[4])
    return pool[0][0]


# =============================================================================
# Bomb value scoring
# =============================================================================

def _bomb_value(my_r, my_c, my_radius, grid, players, agent_id, danger_any):
    """
    Score a potential bomb at (my_r, my_c).
    Returns (score, hits_box, hits_enemy).
    score = boxes_destroyed * 1 + enemies_hit * 3
    Penalty: -2 if fewer than 2 safe open tiles remain adjacent (tight corridor).
    """
    blast     = _blast_tiles(my_r, my_c, my_radius, grid)
    boxes     = sum(1 for r, c in blast if grid[r, c] == BOX)
    enemies   = sum(
        1 for i in range(len(players))
        if i != agent_id
        and int(players[i][2]) == 1
        and (int(players[i][0]), int(players[i][1])) in blast
    )
    h, w = grid.shape
    adj_safe = sum(
        1 for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
        if (0 <= my_r+dr < h and 0 <= my_c+dc < w
            and grid[my_r+dr, my_c+dc] == GRASS
            and (my_r+dr, my_c+dc) not in danger_any)
    )
    corridor_pen = 2 if adj_safe <= 1 else 0
    score = boxes * 1 + enemies * 3 - corridor_pen
    return score, boxes > 0, enemies > 0


# =============================================================================
# Enemy proximity helpers
# =============================================================================

def _enemy_adjacent(my_r, my_c, players, agent_id):
    """True if any live enemy with bombs_left > 0 is adjacent (Manhattan <= 2)."""
    for i, p in enumerate(players):
        if i == agent_id:
            continue
        if int(p[2]) != 1:
            continue
        if int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) <= 2:
            return True
    return False


def _count_nearby_armed(my_r, my_c, players, agent_id):
    """Count live enemies with bombs_left > 0 within Manhattan 2."""
    count = 0
    for i, p in enumerate(players):
        if i == agent_id:
            continue
        if int(p[2]) != 1:
            continue
        if int(p[3]) <= 0:
            continue
        er, ec = int(p[0]), int(p[1])
        if abs(er - my_r) + abs(ec - my_c) <= 2:
            count += 1
    return count


# =============================================================================
# Agent
# =============================================================================

class Agent:
    def __init__(self, agent_id: int):
        self.agent_id    = int(agent_id)
        self.step_count  = 0
        # C4: history for stuck detection and anti-repeat
        self.pos_history = deque(maxlen=20)
        self.last_pos    = None  # position at previous step (for anti-repeat)

    def act(self, obs: dict) -> int:
        try:
            result = self._act_impl(obs)
            return int(result)
        except Exception:
            return STOP
        finally:
            self.step_count += 1

    def _act_impl(self, obs: dict) -> int:
        t0 = time.perf_counter()

        grid    = obs["map"]
        players = obs["players"]
        bombs_a = obs["bombs"]

        # My state
        me = players[self.agent_id]
        my_r, my_c   = int(me[0]), int(me[1])
        alive        = int(me[2])
        bombs_left   = int(me[3])
        radius_bonus = int(me[4])

        if not alive:
            return STOP

        my_radius = max(1, min(MAX_RADIUS, 1 + radius_bonus))
        pos       = (my_r, my_c)

        # C4: update position history; capture prev_pos for anti-repeat
        self.pos_history.append(pos)
        prev_pos      = self.last_pos  # where we were last step
        self.last_pos = pos            # will be prev_pos on next step

        # Timer-aware danger map
        danger_by_time, danger_any = _build_danger_timed(obs)

        # Bomb positions
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs_a}

        # C2/C3: combat mode detection
        nearby_armed = _count_nearby_armed(my_r, my_c, players, self.agent_id)
        combat_mode  = nearby_armed >= 1

        # C1: predict enemy blasts (only when in combat_mode)
        extra_danger = set()
        if combat_mode:
            extra_danger = _enemy_predicted_blast(
                my_r, my_c, players, self.agent_id, grid)

        # C4a: stuck detection (farm_mode only -- no bombs on map, not in danger)
        is_stuck = (
            len(self.pos_history) >= 16
            and len(set(self.pos_history)) <= 3
            and len(bombs_a) == 0
            and pos not in danger_any
        )

        # PRIORITY 1: ESCAPE
        in_immediate = pos in danger_by_time.get(1, set())
        in_any       = pos in danger_any

        if in_immediate or in_any:
            return _escape_timed(pos, grid, bomb_pos, danger_by_time, danger_any)

        # PRIORITY 2: PICK UP ITEM
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w = grid.shape
            items = {(r, c) for r in range(h) for c in range(w)
                     if grid[r, c] in (ITEM_RADIUS, ITEM_CAPACITY)}
            if items:
                a = _bfs_timed(pos, items, grid, bomb_pos, danger_by_time)
                if a is not None and a != STOP:
                    return a

        # C4b: UNSTUCK BOMB (farm_mode only -- never in combat_mode)
        if is_stuck and not combat_mode and bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                blast   = _blast_tiles(my_r, my_c, my_radius, grid)
                has_box = any(grid[r][c] == BOX for r, c in blast)
                if has_box and _can_escape_after_bomb(
                        my_r, my_c, my_radius, grid, bomb_pos, danger_by_time,
                        extra_danger_tiles=None):
                    return PLACE_BOMB

        # PRIORITY 3: PLACE BOMB (scored + threshold + C1/C2/C3 gates)
        if bombs_left > 0 and pos not in bomb_pos:
            if time.perf_counter() - t0 < TIME_BUDGET_S:
                bval, hits_box, hits_enemy = _bomb_value(
                    my_r, my_c, my_radius, grid, players, self.agent_id, danger_any)

                # C2: raise threshold in combat cluster (>=2 armed enemies nearby)
                threshold = BOMB_COMBAT_CLUSTER_SCORE if nearby_armed >= 2 \
                            else BOMB_MIN_SCORE

                if bval >= threshold and (hits_box or hits_enemy):
                    # C1: escape gate includes predicted enemy blasts
                    can_esc = _can_escape_after_bomb(
                        my_r, my_c, my_radius, grid, bomb_pos, danger_by_time,
                        extra_danger_tiles=extra_danger)

                    # C3: in combat_mode require >=2 distinct escape first-moves
                    if can_esc and combat_mode:
                        can_esc = _count_escape_first_moves(
                            my_r, my_c, my_radius, grid, bomb_pos, danger_by_time,
                            extra_danger_tiles=extra_danger) >= 2

                    if can_esc:
                        return PLACE_BOMB

        # PRIORITY 4: FARM BOXES -- navigate to box-adjacent safe spot
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            h, w = grid.shape
            box_spots = set()
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != BOX:
                        continue
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and grid[nr, nc] not in (WALL, BOX)
                                and (nr, nc) not in bomb_pos
                                and (nr, nc) not in danger_any):
                            box_spots.add((nr, nc))

            if box_spots and pos not in box_spots:
                a = _bfs_timed(pos, box_spots, grid, bomb_pos, danger_by_time)
                if a is not None and a != STOP:
                    return a

        # PRIORITY 5: CHASE NEAREST ENEMY (skip if dangerous)
        if time.perf_counter() - t0 < TIME_BUDGET_S:
            if not _enemy_adjacent(my_r, my_c, players, self.agent_id):
                enemies = {
                    (int(players[i][0]), int(players[i][1]))
                    for i in range(len(players))
                    if i != self.agent_id and int(players[i][2]) == 1
                }
                if enemies:
                    a = _bfs_timed(pos, enemies, grid, bomb_pos, danger_by_time)
                    if a is not None and a != STOP:
                        return a

        # FALLBACK: anti-corner safe move (C4c: pass prev_pos for anti-repeat)
        return _safe_fallback(pos, grid, bomb_pos, danger_by_time,
                              self.step_count, self.agent_id, last_pos=prev_pos)
