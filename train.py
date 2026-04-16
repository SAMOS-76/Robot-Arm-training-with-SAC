import gymnasium as gym
from environments.s0_100_env import S0100Env
from SAC_agent import SACAgent
import os
import torch

def make_env():
    def _init():
        env = S0100Env(render_mode=None)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        # Add any other single-env wrappers here
        return env
    return _init

def main():
    models_dir = "models/SAC"
    log_dir = "SAC_logs_with_rand_sphere"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 4 

    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SACAgent(env, device=device)
    model.train(models_dir)

    env.close()

if __name__ == '__main__':
    main()