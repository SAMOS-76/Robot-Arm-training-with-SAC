import gymnasium as gym
from stack_block_env import S0100Env
from stack_block_env import FrameStack
from SAC_agent_HER import SACAgent
import os
import argparse
import torch

def make_env():
    def _init():
        env = S0100Env(render_mode=None)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        #env = FrameStack(env, num_stack=3)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-step", type=int, default=None, help="Load actor/critic checkpoint from this timestep")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Number of additional training steps to run")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval")
    args = parser.parse_args()

    models_dir = "models/SAC_stacking"
    log_dir = "SAC_logs_with_rand_sphere"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 4 

    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SACAgent(env, device=device, timesteps=args.timesteps, use_images=False)

    start_timestep = 0
    if args.resume_step is not None:
        model.load_checkpoint(models_dir, args.resume_step)
        start_timestep = args.resume_step

    model.train(models_dir, save_timesteps=args.save_timesteps, start_timestep=start_timestep)

    env.close()

if __name__ == '__main__':
    main()