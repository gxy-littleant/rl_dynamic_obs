from __future__ import annotations

from operator import add,sub

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal
from minigrid.minigrid_env import MiniGridEnv

from minigrid.core.world_object import Point, WorldObj

import math
import numpy as np


from typing import Union, Dict, List, Tuple, TypeVar, Optional

from scipy.spatial import KDTree



class DynamicObstaclesEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room with moving obstacles.
    The goal of the agent is to reach the green goal square without colliding
    with any obstacle. A large penalty is subtracted if the agent collides with
    an obstacle and the episode finishes. This environment is useful to test
    Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in
    Partial Observability.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure. A '-1' penalty is
    subtracted if the agent collides with an obstacle.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent collides with an obstacle.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Dynamic-Obstacles-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-8x8-v0`
    - `MiniGrid-Dynamic-Obstacles-16x16-v0`

    """

    def __init__(
        self,
        size=9,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_obstacles=1,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)


        # 使当前大小的最外层变为wall，并没有向外扩展一层
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            # 障碍物放置随机，x，y的范围在 （width，height）
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos
        
    # 自定义放置障碍物函数：
    # 初始时放在右上角，每次往右下角移动，
    # 若出界，则恢复至右上角，重复。
    def _place_obj(self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """
        
        if size is None:
            size = (8, 8)
            # size = (self.grid.width, self.grid.height)

        # size=（5，5）时， 为右下角 5*5 区域
        # 初始化
        if top is None:
            # top = (0, 0)
            top = (self.grid.width - size[0], self.grid.height - size[1])
        else:
            # 非初始位置
            top = (max(top[0], 0), max(top[1], 0))

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")
            
            

            num_tries += 1

            # 第一次放置障碍物，在右下角区域内随机初始化
            if obj.cur_pos is None:
                # print("初始化障碍物位置\n")
                pos = (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                    # self._rand_int(top[0], self.grid.width - 1),
                    # self._rand_int(top[1], self.grid.height - 1),
                )
            else :
                # 已初始化过
                x = obj.cur_pos[0]
                y = obj.cur_pos[1]
                # 超出边界，重新回到右下角
                if x - 1 <= 0  or y - 1 <= 0:
                    # print("越界，回到初始化位置\n")
                    pos = (
                        self._rand_int(self.grid.width - size[0], self.grid.width -2),
                        self._rand_int(self.grid.height - size[1], self.grid.height - 2),
                    )
                else :
                    # 则继续运动
                    # print("持续移动\n")
                    x = obj.cur_pos[0]
                    y = obj.cur_pos[1]
                    pos = (x - 1, y - 1)
                
            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos
    
    """
    增加障碍物隐形边界
    """

    def l2(self, start: np.ndarray, goal: np.ndarray) -> int:
        return int(np.linalg.norm(start - goal, np.inf))  # L2 norm

    """
    Check whether the nearest static obstacle is within radius
    """
    # grid_pos:智能体当前位置的周围邻居
    # walls:[(x1,y1),(x2,y2),(x3,y3)]
    def safe_static(
        self, walls: List, grid_pos: np.ndarray, robot_radius: int
    ) -> bool:
        static_walls = KDTree(np.array(walls))
        _, nn = static_walls.query(grid_pos)
        return self.l2(grid_pos, static_walls.data[nn]) > robot_radius

    
    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(sub, old_pos, (1, 1)))

            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100)
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        direct = [[1,0],[-1,0],[0,1],[0,-1]]
        neighbors = []
        for i in range(4):
            new_x = self.agent_pos[0] + direct[i][0]
            new_y = self.agent_pos[1] + direct[i][1]

            if new_x < 1 or new_x > self.grid.height - 1 or new_y < 1 or new_y > self.grid.width - 1 :
                continue
            if self.obstacles[0].cur_pos[0] == new_x and self.obstacles[0].cur_pos[1] == new_y :
                continue
            neighbors.append([new_x, new_y])

        # walls: List, grid_pos: np.ndarray, robot_radius: int
        # 智能体邻居距离障碍物的距离 < robot_radius，则危险 返回False, 结束
        if self.safe_static([self.obstacles[0].cur_pos], neighbors, 2) == False:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

