import gym
from gym.envs.registration import register
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class FourEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):

        self.height = 11
        self.width = 11
        self.grid = np.zeros((self.height, self.width))
        self.reset_type = 0                             # 0 -> random in room 1, 1 -> center of room 4
        self.goal = [8, 8] #[6, 8]
        
        self.moves = {  0 : [-1, 0],    # UP
                        1 : [0, -1],    # LEFT
                        2 : [1, 0],     # DOWN
                        3 : [0, 1]      # RIGHT
                        }
        self.option_size = len(self.moves) + 8
        self.action_space = spaces.Discrete(len(self.moves) + 8)    # 4 primitive actions + 2 multi-step options
        self.observation_space = spaces.Tuple((spaces.Discrete(self.height), spaces.Discrete(self.width)))

        # Rooms -> [start state, end state), end state is upper limit (not to be included)
        self.rooms = {  0 : [[0, 0], [5, 5]],    
                        1 : [[0, 6], [6, 11]],
                        2 : [[7, 6], [11,11]],
                        3 : [[6, 0], [11, 5]]    }
        self.start_room = 0                             

        # Boundaries, hallways and goals
        self.grid[5, 0:5]   = -1
        self.grid[6, 6:11]  = -1
        self.grid[0:11, 5]  = -1

        self.hallways = { 0 : [[2, 5], [5, 1]],
                          1 : [[6, 8], [2, 5]],
                          2 : [[9, 5], [6, 8]],
                          3 : [[5, 1], [9, 5]]    }
        
        self.pre_hallways = {   0 : [[2, 4], [4, 1]],
                                1 : [[5, 8], [2, 6]],
                                2 : [[9, 6], [7, 8]],
                                3 : [[6, 1], [9, 4]]    }

        for i in range(len(self.hallways)):
            self.grid[self.hallways[i][0][0], self.hallways[i][0][1]] = 0
            
        self.grid[self.goal[0], self.goal[1]] = 1
        self.done = False
        self.reset()                             
                                     
    def reset(self):
        if self.reset_type == 0:                        # resets to random position in room 1
            x = np.random.randint(self.rooms[self.start_room][0][0], self.rooms[self.start_room][1][0])
            y = np.random.randint(self.rooms[self.start_room][0][1], self.rooms[self.start_room][1][1])
        else:                                           # resets to center of room 4
            x = (int)((self.rooms[3][0][0] + self.rooms[3][1][0] - 1) / 2)
            y = (int)((self.rooms[3][0][1] + self.rooms[3][1][1] - 1) / 2)
        self.S =  [x, y]
        self.done = False

    def get_room_hallway(self):                         # returns False -> hallways, True -> room cells
        for i in range(len(self.hallways)):
            if self.S == self.hallways[i][0]:
                return False, i
        for i in range(len(self.rooms)):
            low = self.rooms[i][0]
            high = self.rooms[i][1]
            if self.S[0] >= low[0] and self.S[0] < high[0] and self.S[1] >= low[1] and self.S[1] < high[1]:
                return True, i                
    
    def step(self, action):     # returns reward & changes 'done'
        
        if np.random.randint(0, 3) == 1:
            
            action_list = []            
            for j in range(len(self.moves)):
                if j != action:
                    action_list.append(j)
                    
            action = action_list[np.random.randint(0, 3)]
            
        new_state = [self.S[0] + self.moves[action][0], self.S[1] + self.moves[action][1]]
        # Off-the-grid state
        new_state =  [max(0, new_state[0]), max(0, new_state[1])]
        new_state = [min(new_state[0], self.height - 1), min(new_state[1], self.width - 1)]
        if new_state == self.S:            
            return 0
        
        # when new_state is a boundary position
        if self.grid[new_state[0], new_state[1]] == -1:
            return 0

        # updating state
        self.S = new_state

        # reached goal state
        if self.S == self.goal:
            self.done = True
            return 1      
        return 0


register(
    id='four_room-v0',
    entry_point='four_room_env:FourEnv'
)       
