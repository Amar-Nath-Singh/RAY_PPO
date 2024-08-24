# /usr/bin/python3

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from scipy.integrate import solve_ivp
from kcs import KCS_ode
from gymnasium import spaces
from utils import *

class MultiAgentShipEnvironment(MultiAgentEnv):
    def __init__(self, config: dict):
        super().__init__()
        self.num_agents = config.get('num_agents', 1)
        self.wind_flag = config.get('wind_flag', 0)
        self.wind_speed = config.get('wind_speed', 0)
        self.wind_dir = config.get('wind_dir', 0)
        self.wave_flag = config.get('wave_flag', 0)
        self.wave_height = config.get('wave_height', 0)
        self.wave_period = config.get('wave_period', 0)
        self.wave_dir = config.get('wave_dir', 0)
        self.max_steps = config.get('max_steps', 200)
        self.render_mode = config.get('render_mode', None)

        self.action_space = spaces.Box(low=-0.61, high=0.61, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-100, -np.pi, -100, -1], dtype=np.float64),
                                            high=np.array([100, np.pi, 100, 1], dtype=np.float64))

        self.goal_threshold = 0.5
        self.count_step = 0
        self.episode_count = 0

        self._agent_ids = list(range(self.num_agents))

        self.terminateds = set()
        self.truncateds = set()
        self.save_list = np.zeros((self.max_steps, self.num_agents, 12), dtype=np.float64)
        self.reset()

    def init_agent(self, index):
        region_width = 100 // self.num_agents
        start_x = np.random.rand() * region_width + index * region_width
        start_y = np.random.rand() * 20
        min_goal_distance = 8
        max_goal_distance = 18
        goal_distance = np.random.uniform(min_goal_distance, max_goal_distance)
        goal_angle = np.random.uniform(0, 2 * np.pi)
        goal_x = start_x + goal_distance * np.cos(goal_angle)
        goal_y = start_y + goal_distance * np.sin(goal_angle)
        initial_state = np.array([1, 0, 0, start_x, start_y, np.random.uniform(-np.pi, np.pi), 0, 115.5 / 60, start_x, start_y, goal_x, goal_y], dtype=np.float64)
        return initial_state
    
    def update_agent_state(self, agent_id, yaw_err):
        tspan = (0, 0.3)
        agent = self.agents[agent_id]
        yinit = agent[:7]

        # PD Controller
        yaw = agent[5]
        r = agent[2]
        delta_c = np.clip(3.5 * yaw_err - 4.0 * r, -np.radians(35), np.radians(35))

        sol = solve_ivp(lambda t, v: KCS_ode(t, v, delta_c), tspan, yinit, t_eval=tspan, dense_output=True)
        agent[:7] = np.array(sol.y, dtype=np.float64).T[-1]
        agent[5] = ssa(agent[5])

        return agent

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.terminateds.clear()
        self.truncateds.clear()
        self.count_step = 0

        if self.episode_count % 10 == 0:
            np.save(f'saved_npy/EP_{self.episode_count}', self.save_list)

        print(f"End of Episode {self.episode_count}")
        self.save_list = np.zeros((self.max_steps, self.num_agents, 12), dtype=np.float64)

        self.agents = np.array([self.init_agent(agent_id) for agent_id in self._agent_ids])

        observations_dict = {}
        info_dict = {}
        for agent_id in self._agent_ids:
            observations_dict[agent_id], _, _, _, info_dict[agent_id] = self.getAgent(agent_id)
        return observations_dict, info_dict

    def getAgent(self, agent_id):
        agent = self.agents[agent_id]
        # Cross Track Error and Course Angle Error Calculations
        x_init, y_init, x_goal, y_goal = agent[8], agent[9], agent[10], agent[11]
        x, y, yaw, u, v, r = agent[3], agent[4], agent[5], agent[0], agent[1], agent[2]

        vec1 = np.array([x_goal - x_init, y_goal - y_init], dtype=np.float64)
        vec2 = np.array([x_goal - x, y_goal - y], dtype=np.float64)
        vec1_hat = vec1 / (np.linalg.norm(vec1) + 1e-8)  # Avoid division by zero
        cross_track_error = np.cross(vec2, vec1_hat)
        x_dot = u * np.cos(yaw) - v * np.sin(yaw)
        y_dot = u * np.sin(yaw) + v * np.cos(yaw)

        Uvec = np.array([x_dot, y_dot], dtype=np.float64)
        Uvec_hat = Uvec / (np.linalg.norm(Uvec) + 1e-8)  # Avoid division by zero
        vec2_hat = vec2 / (np.linalg.norm(vec2) + 1e-8)  # Avoid division by zero

        course_angle = np.arctan2(Uvec[1], Uvec[0])
        psi_vec2 = np.arctan2(vec2[1], vec2[0])
        course_angle_err = course_angle - psi_vec2
        course_angle_err = ssa(course_angle_err)

        angle_btw23 = np.arccos(np.clip(np.dot(vec2_hat, Uvec_hat), -1.0, 1.0))
        angle_btw12 = np.arccos(np.clip(np.dot(vec1_hat, vec2_hat), -1.0, 1.0))

        distance_to_waypoint = distance(x - x_goal, y - y_goal)

        R1 = 2 * np.exp(-0.08 * (cross_track_error ** 2)) - 1.0
        R2 = 1.3 * np.exp(-10.0 * abs(course_angle_err)) - 0.3
        R3 = -0.25 * distance_to_waypoint

        observations = np.array([cross_track_error, course_angle_err, distance_to_waypoint, r])
        total_reward = R1 + R2 + R3 
        terminated = False
        truncated = False

        info = {'episode_count': self.episode_count}

        if distance_to_waypoint <= self.goal_threshold or (angle_btw12 > np.pi / 2 and angle_btw23 > np.pi / 2):
            terminated = True

        return observations, total_reward, terminated, truncated, info

    def step(self, actions: dict):
        observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict = {}, {}, {}, {}, {}

        for agent_id, action in actions.items():
            self.agents[agent_id] = self.update_agent_state(agent_id, action)

            if np.any(np.isnan(self.agents[agent_id])):
                print(f"NaN detected in agent {agent_id} state after update: {self.agents[agent_id]}")

        for agent_id in actions.keys():
            observation, reward, terminated, truncated, info = self.getAgent(agent_id)

            if terminated:
                self.terminateds.add(agent_id)
            if truncated:
                self.truncateds.add(agent_id)

            observation_dict[agent_id] = observation
            reward_dict[agent_id] = reward
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated
            info_dict[agent_id] = info

        terminated_dict["__all__"] = len(self.terminateds) == self.num_agents
        truncated_dict["__all__"] = self.count_step >= self.max_steps - 1

        self.save_list[self.count_step] = self.agents.copy()
        self.count_step += 1

        return observation_dict, reward_dict, terminated_dict, truncated_dict, info_dict