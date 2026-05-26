"""
Bomberland Agent
================
Phase routing:
  4 / 3 alive  →  rule-based  (item farm, box clear, bomb enemy)
  2 alive      →  MCTS 1-v-1  (UCB tree + heuristic rollout)

State format (obs):
  map     : np.ndarray (13,13)   0=grass 1=wall 2=box 3=rad_item 4=cap_item
  players : np.ndarray (4,5)     [row, col, alive, bombs_left, bomb_bonus]
  bombs   : np.ndarray (N,4)     [row, col, timer, owner_id]
"""

import random
import time
import math
from collections import deque
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight 2-player simulator used during MCTS rollouts
# ─────────────────────────────────────────────────────────────────────────────

class SimState:
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, grid, players, bombs):
        self.grid    = grid      # np.ndarray 13×13
        self.players = players   # list[list[int]]  [r, c, alive, bombs_left, bonus]
        self.bombs   = bombs     # list[list[int]]  [r, c, timer, owner]

    def copy(self):
        s = object.__new__(SimState)
        s.grid    = self.grid.copy()
        s.players = [p[:] for p in self.players]
        s.bombs   = [b[:] for b in self.bombs]
        return s

    # ── geometry ──────────────────────────────────────────────────────────────

    def passable(self, x, y):
        return 0 <= x < 13 and 0 <= y < 13 and int(self.grid[x, y]) in (0, 3, 4)

    def blast_tiles(self, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not (0 <= x < 13 and 0 <= y < 13):
                    break
                c = int(self.grid[x, y])
                if c == 1:
                    break
                tiles.add((x, y))
                if c == 2:
                    break
        return tiles

    def danger_tiles(self):
        """Return (danger_now, danger_soon).
           danger_now  – blast zones of bombs with timer ≤ 2 (explode this/next step)
           danger_soon – all active blast zones
        """
        now, soon = set(), set()
        for b in self.bombs:
            r, c, t, oid = b
            if t <= 0:
                continue
            radius = max(1, self.players[oid][4] + 1) if 0 <= oid < len(self.players) else 2
            blast  = self.blast_tiles(r, c, radius)
            soon  |= blast
            if t <= 2:
                now |= blast
        return now, soon

    # ── one step ──────────────────────────────────────────────────────────────

    def step(self, actions: dict) -> dict:
        """Simulate one game step.  actions = {agent_id: action_int (0-5)}"""
        rewards = {aid: 0.0 for aid in actions}

        # 1. Movement
        existing_bp = {(b[0], b[1]) for b in self.bombs}
        new_pos = {}
        for aid, a in actions.items():
            if self.players[aid][2] != 1:
                continue
            cx, cy = self.players[aid][0], self.players[aid][1]
            if a in (1, 2, 3, 4):
                dx, dy = self.MOVES[a]
                nx, ny = cx + dx, cy + dy
                new_pos[aid] = ((nx, ny) if self.passable(nx, ny)
                                and (nx, ny) not in existing_bp else (cx, cy))
            else:
                new_pos[aid] = (cx, cy)

        for aid, (nx, ny) in new_pos.items():
            self.players[aid][0], self.players[aid][1] = nx, ny

        # 2. Item collection
        for aid in new_pos:
            if self.players[aid][2] != 1:
                continue
            x, y = self.players[aid][0], self.players[aid][1]
            c = int(self.grid[x, y])
            if c == 3:
                self.players[aid][4] = min(self.players[aid][4] + 1, 4)
                self.grid[x, y]       = 0
                rewards[aid]         += 3.0
            elif c == 4:
                self.players[aid][3] = min(self.players[aid][3] + 1, 5)
                self.grid[x, y]       = 0
                rewards[aid]         += 3.0

        # 3. Bomb placement
        cur_bp = {(b[0], b[1]) for b in self.bombs}
        for aid, a in actions.items():
            if self.players[aid][2] != 1:
                continue
            if a == 5:
                x, y = self.players[aid][0], self.players[aid][1]
                if self.players[aid][3] > 0 and (x, y) not in cur_bp:
                    self.bombs.append([x, y, 7, aid])
                    self.players[aid][3] -= 1
                    cur_bp.add((x, y))

        # 4. Timer countdown
        for b in self.bombs:
            b[2] -= 1

        # 5. Explosions with chain reactions
        exploded, all_blast = set(), set()
        queue = [b for b in self.bombs if b[2] <= 0]
        iters  = 0
        while queue and iters < 25:
            iters += 1
            nxt = []
            for b in queue:
                key = (b[0], b[1])
                if key in exploded:
                    continue
                exploded.add(key)
                owner  = b[3]
                radius = (max(1, self.players[owner][4] + 1)
                          if 0 <= owner < len(self.players) else 2)
                blast  = self.blast_tiles(b[0], b[1], radius)
                all_blast |= blast
                for ob in self.bombs:
                    ok = (ob[0], ob[1])
                    if ok not in exploded and ob[2] > 0 and ok in blast:
                        ob[2] = 0
                        nxt.append(ob)
            queue = nxt

        # 6. Cleanup bombs, restore capacity
        surviving = []
        for b in self.bombs:
            if b[2] <= 0:
                owner = b[3]
                if 0 <= owner < len(self.players):
                    self.players[owner][3] = min(self.players[owner][3] + 1, 5)
            else:
                surviving.append(b)
        self.bombs = surviving

        # 7. Destroy boxes
        for x, y in all_blast:
            if 0 <= x < 13 and 0 <= y < 13 and int(self.grid[x, y]) == 2:
                self.grid[x, y] = 0

        # 8. Kill agents
        for aid in actions:
            if self.players[aid][2] != 1:
                continue
            x, y = self.players[aid][0], self.players[aid][1]
            if (x, y) in all_blast:
                self.players[aid][2] = 0
                rewards[aid]        -= 500.0
                for oid in actions:
                    if oid != aid and self.players[oid][2] == 1:
                        rewards[oid] += 100.0

        return rewards

    def is_terminal(self):
        return sum(1 for p in self.players if p[2] == 1) <= 1


# ─────────────────────────────────────────────────────────────────────────────
#  MCTS node
# ─────────────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ('visits', 'value', 'children', 'untried', 'parent', 'action')

    def __init__(self, untried, parent=None, action=None):
        self.visits   = 0
        self.value    = 0.0
        self.children = {}           # action_int → _Node
        self.untried  = list(untried)
        random.shuffle(self.untried)
        self.parent   = parent
        self.action   = action

    def ucb(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))


# ─────────────────────────────────────────────────────────────────────────────
#  Main Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    # ══════════════════════════════════════════════════════════════════════════
    #  Entry point
    # ══════════════════════════════════════════════════════════════════════════

    def act(self, obs: dict) -> int:
        if obs["players"][self.agent_id][2] != 1:
            return 0
        alive = sum(1 for p in obs["players"] if p[2] == 1)
        return self._mcts_act(obs) if alive <= 2 else self._rule_act(obs)

    # ══════════════════════════════════════════════════════════════════════════
    #  Rule-based  (opening / 3-player mid-game)
    # ══════════════════════════════════════════════════════════════════════════

    def _rule_act(self, obs: dict) -> int:
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        me       = players[self.agent_id]
        my_pos   = (int(me[0]), int(me[1]))
        bleft    = int(me[3])
        bradius  = max(1, int(me[4]) + 1)
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs}
        enemies  = [(int(p[0]), int(p[1]))
                    for i, p in enumerate(players)
                    if i != self.agent_id and p[2] == 1]
        blocked  = (set(enemies) | bomb_pos) - {my_pos}

        dnow, dsoon = self._danger(grid, bombs, players)

        # ── 1. Escape immediately if in blast zone ─────────────────────────
        if my_pos in dnow:
            return self._escape_bfs(grid, my_pos, blocked, dnow, dsoon)
        if my_pos in dsoon:
            mv = self._bfs_to_safe(grid, my_pos, blocked, dsoon)
            if mv is not None:
                return mv

        # ── 2. Collect items ───────────────────────────────────────────────
        items = {(x, y) for x in range(13) for y in range(13)
                 if grid[x, y] in (3, 4)}
        if items:
            mv = self._bfs(grid, my_pos, items, blocked, dsoon)
            if mv is not None:
                return mv

        # ── 3. Place bomb if it hits an enemy or ≥1 box ───────────────────
        if bleft > 0 and my_pos not in bomb_pos:
            blast   = self._blast_g(grid, my_pos[0], my_pos[1], bradius)
            hit_e   = any(e in blast for e in enemies)
            box_cnt = sum(1 for t in blast if grid[t[0], t[1]] == 2)
            if (hit_e or box_cnt >= 1) and self._can_escape(
                    grid, my_pos, blocked, dsoon, bradius):
                return 5

        # ── 4. Navigate to best box-farming position ───────────────────────
        bspots = self._box_spots(grid, blocked)
        if bspots:
            mv = self._bfs(grid, my_pos, bspots, blocked, dsoon)
            if mv is not None:
                return mv

        # ── 5. Chase nearest enemy ─────────────────────────────────────────
        if enemies:
            mv = self._bfs(grid, my_pos, set(enemies), blocked, dsoon)
            if mv is not None:
                return mv

        # ── 6. Safe random walk ────────────────────────────────────────────
        safe = [a for a in range(1, 5)
                if (self._passable(grid, *self._nxt(my_pos, a))
                    and self._nxt(my_pos, a) not in blocked
                    and self._nxt(my_pos, a) not in dsoon)]
        return random.choice(safe) if safe else 0

    # ══════════════════════════════════════════════════════════════════════════
    #  MCTS  (1-v-1 end-game)
    # ══════════════════════════════════════════════════════════════════════════

    def _mcts_act(self, obs: dict, budget: float = 0.082) -> int:
        grid    = obs["map"]
        players = obs["players"]
        bombs   = obs["bombs"]

        alive_ids = [i for i, p in enumerate(players) if p[2] == 1]
        if self.agent_id not in alive_ids:
            return 0
        if len(alive_ids) == 1:
            return self._rule_act(obs)

        mid = self.agent_id
        eid = next(i for i in alive_ids if i != mid)

        init = SimState(
            grid.copy(),
            [[int(x) for x in players[i]] for i in range(len(players))],
            [[int(x) for x in b] for b in bombs]
        )

        root = _Node(untried=self._legal(init, mid))
        root.visits = 1   # bootstrap UCB denominator

        deadline = time.time() + budget
        while time.time() < deadline:
            node  = root
            state = init.copy()

            # ── Selection ──────────────────────────────────────────────────
            while not state.is_terminal() and not node.untried and node.children:
                best_a         = max(node.children,
                                     key=lambda a: node.children[a].ucb())
                dn, ds         = state.danger_tiles()
                ea             = self._heur_act(state, eid, mid, dn, ds)
                state.step({mid: best_a, eid: ea})
                node           = node.children[best_a]

            # ── Expansion ──────────────────────────────────────────────────
            if not state.is_terminal() and node.untried:
                action         = node.untried.pop()
                dn, ds         = state.danger_tiles()
                ea             = self._heur_act(state, eid, mid, dn, ds)
                state.step({mid: action, eid: ea})
                child          = _Node(untried=self._legal(state, mid),
                                       parent=node, action=action)
                node.children[action] = child
                node           = child

            # ── Rollout ────────────────────────────────────────────────────
            score = self._rollout(state, mid, eid, depth=15)

            # ── Back-propagation ───────────────────────────────────────────
            n = node
            while n is not None:
                n.visits += 1
                n.value  += score
                n = n.parent

        if not root.children:
            return self._rule_act(obs)
        return max(root.children, key=lambda a: root.children[a].visits)

    # ── MCTS helpers ──────────────────────────────────────────────────────────

    def _heur_act(self, state: SimState, aid: int, opp_id: int,
                  dnow: set, dsoon: set) -> int:
        """Cheap heuristic action for the 'enemy' agent during tree traversal."""
        p = state.players[aid]
        if p[2] != 1:
            return 0
        pos      = (p[0], p[1])
        bomb_pos = {(b[0], b[1]) for b in state.bombs}

        # Escape
        if pos in dnow or pos in dsoon:
            for a in random.sample([1, 2, 3, 4], 4):
                nx, ny = self._nxt(pos, a)
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a
            return 0

        # Attack: bomb opponent if in blast and can escape
        opp = state.players[opp_id]
        if opp[2] == 1:
            opp_pos = (opp[0], opp[1])
            r       = max(1, p[4] + 1)
            if p[3] > 0 and pos not in bomb_pos:
                blast = state.blast_tiles(pos[0], pos[1], r)
                if opp_pos in blast and self._has_esc_s(state, pos, dsoon | blast):
                    return 5

        # Move toward opponent
        if opp[2] == 1:
            dx, dy = opp[0] - pos[0], opp[1] - pos[1]
            cands  = ([2 if dx > 0 else 1, 4 if dy > 0 else 3]
                      if abs(dx) >= abs(dy)
                      else [4 if dy > 0 else 3, 2 if dx > 0 else 1])
            for a in cands:
                nx, ny = self._nxt(pos, a)
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a

        return random.choice([0, 1, 2, 3, 4])

    def _rollout(self, state: SimState, mid: int, eid: int,
                 depth: int = 15) -> float:
        state    = state.copy()
        total    = 0.0
        disc     = 1.0

        for _ in range(depth):
            if state.is_terminal():
                break
            if state.players[mid][2] != 1 or state.players[eid][2] != 1:
                break
            dn, ds = state.danger_tiles()
            ma     = self._rpol(state, mid, eid, dn, ds)
            ea     = self._rpol(state, eid, mid, dn, ds)
            rw     = state.step({mid: ma, eid: ea})
            total += disc * rw.get(mid, 0.0)
            disc  *= 0.95

        my_a = state.players[mid][2] == 1
        en_a = state.players[eid][2] == 1
        if   my_a and not en_a:  total += disc * 600.0
        elif not my_a and en_a:  total -= disc * 400.0
        elif my_a and en_a:      total += disc * self._eval(state, mid, eid)

        return total

    def _rpol(self, state: SimState, aid: int, oid: int,
              dnow: set, dsoon: set) -> int:
        """Rollout policy: escape > item > bomb enemy > safe move."""
        p = state.players[aid]
        if p[2] != 1:
            return 0
        pos      = (p[0], p[1])
        bomb_pos = {(b[0], b[1]) for b in state.bombs}

        # Escape
        if pos in dnow or pos in dsoon:
            for a in random.sample([1, 2, 3, 4], 4):
                nx, ny = self._nxt(pos, a)
                if state.passable(nx, ny) and (nx, ny) not in dsoon:
                    return a
            return 0

        # Grab nearby item
        for x in range(13):
            for y in range(13):
                if int(state.grid[x, y]) in (3, 4):
                    d0 = abs(x - pos[0]) + abs(y - pos[1])
                    if d0 <= 5:
                        for a in [1, 2, 3, 4]:
                            nx, ny = self._nxt(pos, a)
                            if (state.passable(nx, ny)
                                    and (nx, ny) not in dsoon
                                    and abs(nx - x) + abs(ny - y) < d0):
                                return a

        # Bomb if opponent in blast range
        opp = state.players[oid]
        if opp[2] == 1:
            r = max(1, p[4] + 1)
            if p[3] > 0 and pos not in bomb_pos:
                blast = state.blast_tiles(pos[0], pos[1], r)
                if (opp[0], opp[1]) in blast and self._has_esc_s(state, pos, dsoon | blast):
                    return 5

        safe = [a for a in [1, 2, 3, 4, 0]
                if a == 0 or (state.passable(*self._nxt(pos, a))
                              and self._nxt(pos, a) not in dsoon)]
        return random.choice(safe) if safe else 0

    def _eval(self, state: SimState, mid: int, eid: int) -> float:
        """Heuristic evaluation when rollout ends before terminal."""
        my_p, en_p = state.players[mid], state.players[eid]
        score = (my_p[4] - en_p[4]) * 4.0 + (my_p[3] - en_p[3]) * 2.0
        for x in range(13):
            for y in range(13):
                if int(state.grid[x, y]) in (3, 4):
                    md = abs(my_p[0] - x) + abs(my_p[1] - y)
                    ed = abs(en_p[0] - x) + abs(en_p[1] - y)
                    score += (ed - md) * 0.5
        return score

    def _legal(self, state: SimState, aid: int) -> list:
        p = state.players[aid]
        if p[2] != 1:
            return [0]
        pos = (p[0], p[1])
        bp  = {(b[0], b[1]) for b in state.bombs}
        acts = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._nxt(pos, a)
            if state.passable(nx, ny) and (nx, ny) not in bp:
                acts.append(a)
        if p[3] > 0 and pos not in bp:
            acts.append(5)
        return acts

    def _has_esc_s(self, state: SimState, pos: tuple, danger: set) -> bool:
        """BFS escape check inside SimState."""
        q, seen = deque([(pos, 0)]), {pos}
        while q:
            p, d = q.popleft()
            if p not in danger and d > 0:
                return True
            if d >= 7:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._nxt(p, a)
                npos   = (nx, ny)
                if npos not in seen and state.passable(nx, ny):
                    seen.add(npos)
                    q.append((npos, d + 1))
        return False

    # ══════════════════════════════════════════════════════════════════════════
    #  Shared geometry / navigation
    # ══════════════════════════════════════════════════════════════════════════

    def _nxt(self, pos: tuple, a: int) -> tuple:
        dx, dy = self.MOVES[a]
        return pos[0] + dx, pos[1] + dy

    def _passable(self, grid, x: int, y: int) -> bool:
        return 0 <= x < 13 and 0 <= y < 13 and int(grid[x, y]) in (0, 3, 4)

    def _blast_g(self, grid, bx: int, by: int, radius: int) -> set:
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not (0 <= x < 13 and 0 <= y < 13):
                    break
                c = int(grid[x, y])
                if c == 1:
                    break
                tiles.add((x, y))
                if c == 2:
                    break
        return tiles

    def _danger(self, grid, bombs, players) -> tuple:
        """Return (danger_now, danger_soon) from actual observation data."""
        dnow, dsoon = set(), set()
        for b in bombs:
            bx, by, t = int(b[0]), int(b[1]), int(b[2])
            oid = int(b[3]) if len(b) > 3 else -1
            if t <= 0:
                continue
            r = max(1, int(players[oid][4]) + 1) if 0 <= oid < len(players) else 2
            blast  = self._blast_g(grid, bx, by, r)
            dsoon |= blast
            if t <= 2:
                dnow |= blast
        return dnow, dsoon

    def _escape_bfs(self, grid, pos, blocked, dnow, dsoon) -> int:
        """BFS to first tile outside danger_soon, avoiding danger_now."""
        q, seen = deque([(pos, None)]), {pos}
        while q:
            p, first = q.popleft()
            if p not in dsoon and first is not None:
                return first
            for a in [1, 2, 3, 4]:
                nx, ny = self._nxt(p, a)
                npos   = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny) or npos in blocked:
                    continue
                if npos in dnow:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        # fallback: any move out of danger_now
        for a in [1, 2, 3, 4]:
            nx, ny = self._nxt(pos, a)
            if (self._passable(grid, nx, ny)
                    and (nx, ny) not in blocked
                    and (nx, ny) not in dnow):
                return a
        return 0

    def _bfs_to_safe(self, grid, pos, blocked, dsoon):
        """BFS toward nearest safe tile (no danger_soon restriction en route)."""
        q, seen = deque([(pos, None)]), {pos}
        while q:
            p, first = q.popleft()
            if p not in dsoon and first is not None:
                return first
            for a in [1, 2, 3, 4]:
                nx, ny = self._nxt(p, a)
                npos   = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny) or npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        return None

    def _bfs(self, grid, start, targets, blocked, avoid) -> int | None:
        """BFS first-move toward any tile in targets, dodging avoid set."""
        if not targets:
            return None
        q, seen = deque([(start, None)]), {start}
        while q:
            pos, first = q.popleft()
            if pos in targets and first is not None:
                return first
            for a in [1, 2, 3, 4]:
                nx, ny = self._nxt(pos, a)
                npos   = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in blocked and npos not in targets:
                    continue
                if npos in avoid:
                    continue
                seen.add(npos)
                q.append((npos, a if first is None else first))
        return None

    def _can_escape(self, grid, pos, blocked, dsoon, radius) -> bool:
        """Check whether agent can exit own bomb blast after placing."""
        blast    = self._blast_g(grid, pos[0], pos[1], radius)
        combined = dsoon | blast
        q, seen  = deque([(pos, 0)]), {pos}
        while q:
            p, d = q.popleft()
            if p not in combined and d > 0:
                return True
            if d >= 8:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._nxt(p, a)
                npos   = (nx, ny)
                if npos in seen or not self._passable(grid, nx, ny) or npos in blocked:
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return False

    def _box_spots(self, grid, blocked) -> set:
        """Tiles adjacent to boxes that are passable and unblocked."""
        spots = set()
        for x in range(13):
            for y in range(13):
                if grid[x, y] != 2:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                        spots.add((nx, ny))
        return spots