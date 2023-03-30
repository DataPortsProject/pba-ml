import gym
from pathlib import Path
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import os

inner_env = None
global_max_episode_number = None


def run(env_string, model=None, policy=MlpPolicy, max_learning_steps=4300, verbose=0, n_steps=128, nminibatches=4,
        gamma=0.99, rl_learning_rate=2.5e-4, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, cliprange=0.2,
        cliprange_vf=None, lam=0.95, policy_kwargs=None, tensorboard_log=None, save_path=None, load_path=None,
        max_episode_number=0, **kwargs):
    if policy_kwargs is None:
        policy_kwargs = dict()

    global global_max_episode_number
    global_max_episode_number = max_episode_number

    global inner_env
    inner_env = gym.make(env_string, **kwargs)
    env = DummyVecEnv([lambda: inner_env])

    if model is None:
        model = PPO2(policy=policy, env=env, verbose=verbose, n_steps=n_steps, nminibatches=nminibatches, gamma=gamma,
                     ent_coef=ent_coef, learning_rate=rl_learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                     cliprange=cliprange, cliprange_vf=cliprange_vf, lam=lam, policy_kwargs=policy_kwargs,
                     tensorboard_log=tensorboard_log)
    else:
        model.set_env(env)
    if load_path is not None and load_path != '':
        model = PPO2.load(load_path, env,
                          dict(verbose=verbose, n_steps=n_steps, nminibatches=nminibatches,
                               gamma=gamma,
                               ent_coef=ent_coef, learning_rate=rl_learning_rate, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm,
                               cliprange=cliprange, cliprange_vf=cliprange_vf, lam=lam, policy_kwargs=policy_kwargs,
                               tensorboard_log=tensorboard_log))

    inner_env.model = model
    model.learn(total_timesteps=max_learning_steps, tb_log_name=os.path.basename(__file__).rstrip(".py"),
                callback=tensorboard_callback)
    if save_path is not None and save_path != '':
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    env.close()

    return model


def tensorboard_callback(locals_, globals_):
    global inner_env
    global global_max_episode_number
    self_ = locals_['self']
    if inner_env.summary_writer is None:
        inner_env.summary_writer = locals_['writer']
    if global_max_episode_number != 0 and inner_env.episode_count >= global_max_episode_number:
        return False
    if inner_env.abort:
        return False
    return True


if __name__ == "__main__":
    run(2000000)
