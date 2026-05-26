import random
from collections import deque

class SamnuAgent:
    """
    Actions:
    0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    team_id = "Samnu"
        
    def __init__(self, agent_id : int):
        self.agent_id = agent_id

    """
    obs = {
        "map":     np.ndarray,  # shape (13, 13), dtype int
                                # 0=Grass, 1=Wall, 2=Box, 3=Item_Radius, 4=Item_Capacity
        "players": np.ndarray,  # shape (4, 5), dtype int8
                                # Mỗi hàng: [row, col, alive, bombs_left, bomb_radius_bonus]
        "bombs":   np.ndarray,  # shape (N, 4), dtype int8, N = số bom hiện có
                                # Mỗi hàng: [row, col, timer, owner_id]
    }
    """
    
    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]


        
    



    
