from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
# from safe_control_gym.controllers import PID
from safe_control_gym.controllers.lqr.lqr import LQR
# from safe_control_gym.envs.benchmark_env import Task

import heapq

"""
在2slow版本基础上进行步长调整，以及障碍物检测调整
然后加入了3d grid
"""
class AStarNode:
    def __init__(self, position, parent=None, g=0, h=0, f=0):
        self.position = tuple(position)
        self.parent = parent
        self.g = g
        self.h = h
        self.f = f
        
    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    return np.linalg.norm(np.array(a[:3]) - np.array(b[:3]))

def create_3d_grid(grid_size):
    gates=[  # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
            [0.45, -1.0, 0, 0, 0, 2.35, 1],
            [1.0, -1.55, 0, 0, 0, -0.78, 0],
            [0.0, 0.5, 0, 0, 0, 0, 1],
            [-0.5, -0.5, 0, 0, 0, 3.14, 0]
        ]
    min_x = min(gate[0] for gate in gates)
    max_x = max(gate[0] for gate in gates)
    min_y = min(gate[1] for gate in gates)
    max_y = max(gate[1] for gate in gates)
    min_z = min(gate[2] for gate in gates)
    max_z = max(gate[2] for gate in gates)

    grid = []
    for i in np.arange(min_x, max_x + grid_size, grid_size):
        plane = []
        for j in np.arange(min_y, max_y + grid_size, grid_size):
            row = []
            for k in np.arange(min_z, max_z + grid_size, grid_size):
                row.append(AStarNode((i, j, k)))
            plane.append(row)
        grid.append(plane)
    return grid

def astar(start_node, end_node, obstacles):
    grid_size = 0.05
    grid = create_3d_grid(grid_size)
    open_list = []
    closed_list = set()
    # start_node = AStarNode(start_node)
    # end_node = AStarNode(end_node)
    heapq.heappush(open_list, start_node)
    # print("start node in astar func:", start_node)
    

    step_size = 0.05  # Increased step size to reduce the number of nodes

    while open_list:
        # print("open list len: ", len(open_list))
        current_node = heapq.heappop(open_list)
        # print("current node:", current_node)
        closed_list.add(current_node.position)
        print("current_node pos:", current_node.position)

        # if current_node.position == end_node.position: # toooooo strict!
        if np.linalg.norm(np.array(current_node.position) - np.array(end_node.position)) < 0.5:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
                print("path in loop:", path, "\n path type: ", type(path))
            return path[::-1]

        # no grid版本
        # neighbors = [
        #     (current_node.position[0] + dx, current_node.position[1] + dy, current_node.position[2] + dz)
        #     for dx, dy, dz in [
        #         (-step_size, 0, 0), (step_size, 0, 0), (0, -step_size, 0), (0, step_size, 0),
        #         (0, 0, -step_size), (0, 0, step_size)
        #     ]
        # ]

        # grid版本：
        # Get the indices of the current node in the grid
        x, y, z = current_node.position
        ix = int((x - (-0.7)) / grid_size) # x - min_x
        iy = int((y - (-1.75)) / grid_size)
        iz = int((z - 0) / grid_size)

        # Get the neighbors from the grid
        neighbors = []
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and 0 <= nz < len(grid[0][0]):
                neighbors.append(grid[nx][ny][nz])


        for next_position in neighbors:
            if next_position in closed_list or not is_valid_position(next_position, obstacles):
                continue

            g = current_node.g + heuristic(current_node.position, next_position)
            h = heuristic(next_position, end_node.position)
            f = g + h
            neighbor_node = AStarNode(next_position, current_node, g, h, f)

            if add_to_open(open_list, neighbor_node):
                heapq.heappush(open_list, neighbor_node)

    return heapq

def is_valid_position(position, obstacles):
    for obs in obstacles:
        if heuristic(position, obs[:3]) < 0.3:  # Increased threshold for collision detection
            return False
    return True

def add_to_open(open_list, neighbor_node):
    for node in open_list:
        if neighbor_node.position == node.position and neighbor_node.f >= node.f:
            return False
    return True

# 整体寻找
# def find_path(start, end, gates, obstacles):
#     path = []
#     waypoints = [start] + gates + [end]

#     for i in range(len(waypoints) - 1):
#         segment = astar(waypoints[i], waypoints[i + 1], obstacles)
#         if segment is None:
#             print(f"Path segment from {waypoints[i]} to {waypoints[i+1]} not found")
#             return None
#         path.extend(segment[:-1])

#     path.append(end)
#     return path

def plan_path(waypoints, obstacles):
    """
    首先将起点添加到路径中，然后对于每一对连续的门，我们使用astar函数找到了当前门到下一个 \n
    门的路径，并将其添加到总路径中。如果任何时候astar函数返回none，那么满足要求的总路径就不存在
    """
    # path = [start]
    # waypoints = [start] + gates + [end]

    # for i in range(len(waypoints) - 1):
    #     start_node = AStarNode(waypoints[i])
    #     end_node = AStarNode(waypoints[i + 1])
    #     segment = astar(start_node, end_node, obstacles)
    #     if segment is None:
    #         print(f"Path segment from {waypoints[i]} to {waypoints[i+1]} not found")
    #         return None
    #     # Convert nodes back to positions
    #     segment = [node.get_position() for node in segment]
    #     # Remove the first point of the segment because it is the same as the last point of the path
    #     path += segment[1:]

    # return path
    # gate_centers = [[(g[0] + g[2]) / 2, (g[1] + g[3]) / 2, (g[4] + g[5]) / 2] for g in gates]

    # # Add the start, gate centers, and end to the waypoints
    # waypoints = [start] + gate_centers + [end]

    path = []
    for i in range(len(waypoints) - 1):
        start_node = AStarNode(waypoints[i])
        end_node = AStarNode(waypoints[i + 1])
        segment = astar(start_node, end_node, obstacles)
        if segment is None:
            print("No path found")
            return None
        path.extend(segment)

    return path

class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            information contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        initial_pos = [self.initial_obs[0], self.initial_obs[2], 0.3]
        # goal_pos = initial_info["task_info"]["stabilization_goal"]
        goal_pos = [ # 终点， 来自initial_info
            initial_info["x_reference"][0],
            initial_info["x_reference"][2],
            initial_info["x_reference"][4],
        ]
        # gates = [[gate[0], gate[1], gate[2]] for gate in self.NOMINAL_GATES]
        # print("gates: ", gates) # [[0.45, -1.0, 0], [1.0, -1.55, 0], [0.0, 0.5, 0], [-0.5, -0.5, 0]]
        # obstacles = [[obs[0], obs[1], obs[2]] for obs in self.NOMINAL_OBSTACLES]
        gates = [gate[ :3] for gate in self.NOMINAL_GATES] # first 3 dims - x y z
        # print("gates: ", gates) 
        obstacles = [obstacle[ :3] for obstacle in self.NOMINAL_OBSTACLES]

        waypoints = []
        waypoints.append([self.initial_obs[0], self.initial_obs[2], 0.3])
        gates = self.NOMINAL_GATES
        z_low = initial_info["gate_dimensions"]["low"]["height"]
        z_high = initial_info["gate_dimensions"]["tall"]["height"]
        waypoints.append([1, 0, z_low])
        waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
        waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
        waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
        waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.1, z_low])
        waypoints.append([gates[2][0], gates[2][1] + 0.1, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints.append(
            [
                initial_info["x_reference"][0],
                initial_info["x_reference"][2],
                initial_info["x_reference"][4],
            ]
        )
        waypoints = np.array(waypoints)
        print("waypoints:", waypoints)
        print("obstacles:", obstacles)

        path = plan_path(waypoints, [tuple(o) for o in obstacles])
        if path is None:
            raise ValueError("No valid path found")

        path = np.array(path)
        print("path found !!!!!!!!!! : /n", path)
        tck, u = interpolate.splprep([path[:, 0], path[:, 1], path[:, 2]], s=0.1)
        self.path = path
        duration = 12
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.path, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False

        #########################
        # REPLACE THIS (END) ####
        #########################

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.ref_x):
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif step >= len(self.ref_x) and not self._setpoint_land:
                command_type = Command.SETPOINT_STOP
                args = []
                self._setpoint_land = True
            # elif self._setpoint_land and not self._land:
            #     command_type = Command.LAND
            #     args = [self.ref_x[-1], self.ref_y[-1], 0.0, 2]  # X, Y, Z, duration
            #     self._land = True
            elif step >= len(self.ref_x) and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################
