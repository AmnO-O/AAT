import random
from collections import deque


class BoxFarmerAgent:
    """
    Box-and-item focused agent:
    - escape first,
    - collect items,
    - farm boxes,
    - place bombs only if an escape path exists,
    - avoid stepping onto tiles that will die next turn.
    """

    team_id = "BoxFarmerAgent"

    MOVES = {
        0: (0, 0),    # STOP
        1: (-1, 0),   # LEFT
        2: (1, 0),    # RIGHT
        3: (0, -1),   # UP
        4: (0, 1),    # DOWN
    }

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or int(players[self.agent_id][2]) != 1:
            return 0

        my_r, my_c, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_r), int(my_c))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        occupied = {
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and int(p[2]) == 1
        }
        blocked = set(occupied) | bomb_positions
        blocked.discard(my_pos)

        # New explicit next-turn safety planes
        survive_next_turn, die_next_turn = self._next_turn_survival_planes(grid, bombs, players)

        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # 1) Escape immediately if current tile dies next turn or is already in danger.
        if my_pos in die_next_turn or not survive_next_turn[my_pos[0]][my_pos[1]]:
            escape = self._move_to_nearest_safe(
                grid,
                my_pos,
                blocked,
                survive_next_turn,
                search_depth=8,
            )
            if escape is not None:
                return escape

            safe_moves = [
                a for a in valid_actions
                if a != 0 and survive_next_turn[self._next_pos(my_pos, a)[0]][self._next_pos(my_pos, a)[1]]
            ]
            return random.choice(safe_moves) if safe_moves else 0

        # 2) Collect items if reachable and safe next turn.
        item_tiles = self._item_tiles(
            grid,
            prefer_capacity=int(bombs_left) <= 1,
            prefer_radius=int(bomb_bonus) <= 1,
        )
        if item_tiles:
            move = self._move_to_targets(
                grid,
                my_pos,
                item_tiles,
                blocked,
                survive_next_turn,
            )
            if move is not None:
                return move

        # 3) Place bomb only if it will help and we can escape to a tile that survives next turn.
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_here = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            if boxes_here > 0 and self._can_escape_after_placing(
                grid,
                my_pos,
                blocked,
                bombs,
                players,
                bomb_radius,
            ):
                return 5

        # 4) Move toward good bomb spots near boxes.
        box_spots = self._box_bomb_spots(grid, blocked)
        if box_spots:
            move = self._move_to_targets(
                grid,
                my_pos,
                box_spots,
                blocked,
                survive_next_turn,
            )
            if move is not None:
                return move

        # 5) If nothing else, move safely.
        safe_moves = [
            a for a in valid_actions
            if a != 0 and survive_next_turn[self._next_pos(my_pos, a)[0]][self._next_pos(my_pos, a)[1]]
        ]
        return random.choice(safe_moves) if safe_moves else 0

    def _next_pos(self, pos, action):
        dr, dc = self.MOVES[int(action)]
        return pos[0] + dr, pos[1] + dc

    def _in_bounds(self, grid, r, c):
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

    def _passable(self, grid, r, c):
        return self._in_bounds(grid, r, c) and int(grid[r, c]) in (0, 3, 4)

    def _valid_actions(self, grid, my_pos, blocked):
        actions = [0]
        for a in (1, 2, 3, 4):
            nr, nc = self._next_pos(my_pos, a)
            if self._passable(grid, nr, nc) and (nr, nc) not in blocked:
                actions.append(a)
        return actions

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for d in range(1, radius + 1):
                x, y = bx + dr * d, by + dc * d
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _bomb_radius_for_owner(self, players, owner_id):
        if 0 <= owner_id < len(players) and int(players[owner_id][2]) == 1:
            return max(1, 1 + int(players[owner_id][4]))
        return 1

    def _bomb_effective_explosion_tiles_next_turn(self, grid, bombs, players):
        """
        Return the set of tiles that will explode by the next turn,
        including chain reactions from bombs with timer <= 1.
        """
        if bombs is None or len(bombs) == 0:
            return set()

        n = len(bombs)
        scheduled = set()

        # Bombs that already explode next turn.
        for i in range(n):
            timer = int(bombs[i][2])
            if timer <= 1:
                scheduled.add(i)

        changed = True
        while changed:
            changed = False
            blast_union = set()
            for i in scheduled:
                bx, by = int(bombs[i][0]), int(bombs[i][1])
                owner_id = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
                radius = self._bomb_radius_for_owner(players, owner_id)
                blast_union |= self._blast_tiles(grid, bx, by, radius)

            for j in range(n):
                if j in scheduled:
                    continue
                bx, by = int(bombs[j][0]), int(bombs[j][1])
                if (bx, by) in blast_union:
                    scheduled.add(j)
                    changed = True

        danger_tiles = set()
        for i in scheduled:
            bx, by = int(bombs[i][0]), int(bombs[i][1])
            owner_id = int(bombs[i][3]) if bombs.shape[1] > 3 else -1
            radius = self._bomb_radius_for_owner(players, owner_id)
            danger_tiles |= self._blast_tiles(grid, bx, by, radius)

        return danger_tiles

    def _next_turn_survival_planes(self, grid, bombs, players):
        """
        Returns:
            survive_next_turn: bool ndarray (13,13)
            die_next_turn: bool ndarray (13,13)
        """
        danger_tiles = self._bomb_effective_explosion_tiles_next_turn(grid, bombs, players)

        survive_next_turn = [
            [False for _ in range(grid.shape[1])]
            for _ in range(grid.shape[0])
        ]
        die_next_turn = [
            [False for _ in range(grid.shape[1])]
            for _ in range(grid.shape[0])
        ]

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if int(grid[r, c]) in (0, 3, 4):
                    survive_next_turn[r][c] = (r, c) not in danger_tiles
                    die_next_turn[r][c] = (r, c) in danger_tiles

        return survive_next_turn, die_next_turn

    def _danger_tiles(self, grid, bombs, players, default_radius=2):
        danger_soon = set()
        danger_now = set()
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0:
                continue
            radius = default_radius
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            blast = self._blast_tiles(grid, bx, by, radius)
            danger_soon |= blast
            if timer <= 1:
                danger_now |= blast
        return danger_soon, danger_now

    def _move_to_nearest_safe(self, grid, start, occupied, survive_next_turn, search_depth=8):
        q = deque([(start, 0, None)])
        seen = {start}

        while q:
            pos, dist, first_action = q.popleft()

            if dist > 0 and survive_next_turn[pos[0]][pos[1]]:
                return first_action

            if dist >= search_depth:
                continue

            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if npos in occupied:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, dist + 1, a if first_action is None else first_action))

        return None

    def _move_to_targets(self, grid, start, targets, occupied, survive_next_turn):
        if not targets:
            return None

        q = deque([(start, None)])
        seen = {start}

        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action

            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if npos in occupied:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                if not survive_next_turn[npos[0]][npos[1]]:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))

        return None

    def _item_tiles(self, grid, prefer_capacity=False, prefer_radius=False):
        preferred_values = set()
        if prefer_radius:
            preferred_values.add(3)
        if prefer_capacity:
            preferred_values.add(4)

        preferred = {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in preferred_values
        }
        if preferred:
            return preferred

        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if int(grid[x, y]) in (3, 4)
        }

    def _count_boxes_in_blast(self, grid, my_pos, radius):
        return sum(
            1
            for x, y in self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
            if int(grid[x, y]) == 2
        )

    def _can_escape_after_placing(self, grid, my_pos, occupied, bombs, players, bomb_radius):
        """
        Hypothetically add the bomb, then check whether there exists at least one
        adjacent tile that survives next turn.
        """
        hyp_bombs = self._add_hypothetical_bomb(bombs, my_pos, self.agent_id)
        survive_next_turn, _ = self._next_turn_survival_planes(grid, hyp_bombs, players)

        # The current tile may become unsafe after placing; we only care that
        # there is a move away that survives next turn.
        for a in (1, 2, 3, 4):
            npos = self._next_pos(my_pos, a)
            if npos in occupied:
                continue
            if not self._passable(grid, npos[0], npos[1]):
                continue
            if survive_next_turn[npos[0]][npos[1]]:
                return True

        # Fallback BFS: can we reach any surviving tile in a few steps?
        q = deque([(my_pos, 0)])
        seen = {my_pos}
        while q:
            pos, dist = q.popleft()
            if dist > 0 and survive_next_turn[pos[0]][pos[1]]:
                return True
            if dist >= 8:
                continue
            for a in (1, 2, 3, 4):
                npos = self._next_pos(pos, a)
                if npos in seen:
                    continue
                if npos in occupied:
                    continue
                if not self._passable(grid, npos[0], npos[1]):
                    continue
                seen.add(npos)
                q.append((npos, dist + 1))

        return False

    def _add_hypothetical_bomb(self, bombs, pos, owner_id, timer=7):
        row = [pos[0], pos[1], timer, owner_id]
        if bombs is None or len(bombs) == 0:
            return __import__("numpy").array([row], dtype=__import__("numpy").int8)
        import numpy as np
        return np.concatenate([bombs, np.array([row], dtype=np.int8)], axis=0)

    def _box_bomb_spots(self, grid, occupied):
        """
        Return walkable tiles adjacent to boxes.
        These are good places to move to before placing bombs.
        """
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                        spots.add((nx, ny))
        return spots