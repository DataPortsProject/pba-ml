from gym_threshold_begone.envs.bpm2020_env_fixed_processlength import BPM2020EnvFixedProcesslength
from gym_threshold_begone.envs._baseenv import BaseEnv
from python_src.rl_algorithms.ppo2 import train_run, PPO
import gym

import time
import os
import math

BaseEnv.show_graphs = False

BaseEnv.experiment_number = 1
env_wrapper = gym.make("non_prophetic_curiosity-v0", predictive_output_directory="../models-bpic-extracted")
observation_dimensions = env_wrapper.observation_space.shape[0]
num_actions = env_wrapper.action_space.n
ppo = PPO(observation_dimensions=observation_dimensions, num_actions=num_actions)
env_wrapper.env.model = ppo
train_run(env_wrapper, ppo, episodes=4362, save_name="saved_ppo_bpm2020/bpic/models")

BaseEnv.experiment_number = 2
env_wrapper = gym.make("non_prophetic_curiosity-v0", predictive_output_directory="../models-bpic17-extracted")
observation_dimensions = env_wrapper.observation_space.shape[0]
num_actions = env_wrapper.action_space.n
ppo = PPO(observation_dimensions=observation_dimensions, num_actions=num_actions)
env_wrapper.env.model = ppo
train_run(env_wrapper, ppo, episodes=10502, save_name="saved_ppo_bpm2020/bpic17/models")

BaseEnv.experiment_number = 3
env_wrapper = gym.make("non_prophetic_curiosity-v0", predictive_output_directory="../models-c2k-extracted")
observation_dimensions = env_wrapper.observation_space.shape[0]
num_actions = env_wrapper.action_space.n
ppo = PPO(observation_dimensions=observation_dimensions, num_actions=num_actions)
env_wrapper.env.model = ppo
train_run(env_wrapper, ppo, episodes=1314, save_name="saved_ppo_bpm2020/c2k/models")

BaseEnv.experiment_number = 4
env_wrapper = gym.make("non_prophetic_curiosity-v0", predictive_output_directory="../models-traffic-extracted")
observation_dimensions = env_wrapper.observation_space.shape[0]
num_actions = env_wrapper.action_space.n
ppo = PPO(observation_dimensions=observation_dimensions, num_actions=num_actions)
env_wrapper.env.model = ppo
train_run(env_wrapper, ppo, episodes=50119, save_name="saved_ppo_bpm2020/traffic/models")
