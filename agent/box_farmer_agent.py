import random
from collections import deque


class BoxFarmerAgent:
    """
    Box-and-item focused agent:
    - escape first,
    - collect items,
    - farm boxes,
    - place bombs only if an escape path exists.
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

        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)
        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # 1) Escape immediately if in danger.
        if my_pos in danger_now or my_pos in danger_soon:
            escape = self._move_to_nearest_safe(grid, my_pos, blocked, danger_soon, search_depth=8)
            if escape is not None:
                return escape

            safe_moves = [
                a for a in valid_actions
                if a != 0 and self._next_pos(my_pos, a) not in danger_now
            ]
            return random.choice(safe_moves) if safe_moves else 0

        # 2) Collect items if reachable.
        item_tiles = self._item_tiles(
            grid,
            prefer_capacity=int(bombs_left) <= 1,
            prefer_radius=int(bomb_bonus) <= 1,
        )
        if item_tiles:
            move = self._move_to_targets(grid, my_pos, item_tiles, blocked, danger_soon)
            if move is not None:
                return move

        # 3) Place bomb only if it will help and we can escape safely.
        if bombs_left > 0 and my_pos not in bomb_positions:
            boxes_here = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            if boxes_here > 0 and self._can_escape_after_placing(
                grid, my_pos, blocked, danger_soon, bomb_radius
            ):
                return 5

        # 4) Move toward good bomb spots near boxes.
        box_spots = self._box_bomb_spots(grid, blocked)
        if box_spots:
            move = self._move_to_targets(grid, my_pos, box_spots, blocked, danger_soon)
            if move is not None:
                return move

        # 5) If nothing else, move safely.
        safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_soon]
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
                r, c = bx + dr * d, by + dc * d
                if not self._in_bounds(grid, r, c):
                    break
                cell = int(grid[r, c])
                if cell == 1:
                    break
                tiles.add((r, c))
                if cell == 2:
                    break
        return tiles

    def _danger_tiles(self, grid, bombs, players):
        """
        danger_soon: tiles that will explode soon enough that we should avoid them.
        danger_now: tiles that are exploding immediately (timer <= 1).
        """
        danger_soon = set()
        danger_now = set()

        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1

            if timer <= 0:
                continue

            radius = 1
            if 0 <= owner_id < len(players) and int(players[owner_id][2]) == 1:
                radius = max(1, int(players[owner_id][4]) + 1)

            blast = self._blast_tiles(grid, bx, by, radius)
            danger_soon |= blast
            if timer <= 1:
                danger_now |= blast

        return danger_soon, danger_now

    def _move_to_nearest_safe(self, grid, start, occupied, danger_soon, search_depth=8):
        q = deque([(start, 0, None)])
        seen = {start}

        while q:
            pos, dist, first_action = q.popleft()

            if dist > 0 and pos not in danger_soon:
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

    def _move_to_targets(self, grid, start, targets, occupied, danger_soon):
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
                if npos in danger_soon:
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

    def _can_escape_after_placing(self, grid, my_pos, occupied, existing_danger, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        combined_danger = set(existing_danger) | my_blast
        escape = self._move_to_nearest_safe(
            grid, my_pos, occupied, combined_danger, search_depth=8
        )
        return escape is not None

    def _box_bomb_spots(self, grid, occupied):
        """
        Return walkable tiles adjacent to boxes.
        This helps the agent move to useful bomb positions.
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