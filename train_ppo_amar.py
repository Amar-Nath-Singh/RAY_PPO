import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.ppo import PPOConfig
from multi_agent_env import MultiAgentShipEnvironment
from gymnasium import spaces
import numpy as np

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_0"

# Configure the environment and agents
params = {
    "num_agents": 1,
    "wind_flag": 0,
    "wind_speed": 0,
    "wind_dir": 0,
    "wave_flag": 0,
    "wave_height": 0,
    "wave_period": 0,
    "wave_dir": 0,
    "max_steps": 150,
    "render_mode": None
}

# Define the observation and action spaces for the agents
obs_space = spaces.Box(low=np.array([-100, -np.pi, -100, -1], dtype=np.float64), 
                       high=np.array([100, np.pi, 100, 1], dtype=np.float64))

act_space = spaces.Box(low=-np.radians(180), high=np.radians(180), shape=(1,), dtype=np.float64)

# Use PPOConfig to set up the configuration
config = PPOConfig()
config.environment(env = MultiAgentShipEnvironment, env_config=params)
config.framework(framework = "torch")
config.resources(num_gpus = 1)
config.env_runners(num_env_runners = 23)

config.model = {
    "fcnet_hiddens": [128, 128],
    "fcnet_activation": "tanh",
}

config.training(
    clip_param = 0.2,
    lambda_ = 0.95,
    entropy_coeff = 0.001,
)
config.lr = 0.001
config.gamma = 0.95
config.timesteps_per_iteration = 50 * params['max_steps']

# Multi-agent setup
config.multi_agent(
    policies={f"policy_0": (None, obs_space, act_space, {})},
    policy_mapping_fn=policy_mapping_fn,
    policies_to_train=[f"policy_0"]
)

training_iteration = 100
tuner = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(stop={"training_iteration": training_iteration}, 
                                     storage_path='/home/guildstudent/AMAR_ML/RAY_PPO/ray_results'),
            param_space=config.to_dict()
        )

tuner.fit()
ray.shutdown()