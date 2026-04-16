import gymnasium as gym
from stack_block_env import S0100Env
from stack_block_env import FrameStack
from SAC_agent_HER import SACAgent
import os
import argparse
import torch

def make_env(max_episode_steps=400, success_hold_steps=3):
    def _init():
        env = S0100Env(
            render_mode=None,
            max_episode_steps=max_episode_steps,
            success_hold_steps=success_hold_steps,
        )
        # env = FrameStack(env, num_stack=3)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-step", type=int, default=None, help="Load actor/critic checkpoint from this timestep")
    parser.add_argument("--timesteps", type=int, default=30_000_000, help="Number of additional training steps to run")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval")
    parser.add_argument("--episode-steps", type=int, default=400, help="Episode horizon handled by S0100Env")
    parser.add_argument("--success-hold-steps", type=int, default=1, help="Consecutive success steps before terminate")
    parser.add_argument("--num-envs", type=int, default=12, help="Number of parallel environments")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--updates-per-step", type=int, default=2, help="Gradient updates per env step")
    args = parser.parse_args()

    models_dir = "models/SAC_stacking"
    log_dir = "SAC_logs_with_rand_sphere"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = args.num_envs
    env = gym.vector.AsyncVectorEnv([make_env(args.episode_steps, args.success_hold_steps) for _ in range(num_envs)])
    model = SACAgent(
        env,
        device=device,
        timesteps=args.timesteps,
        use_images=False,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
    )

    start_timestep = 0
    if args.resume_step is not None:
        model.load_checkpoint(models_dir, args.resume_step)
        start_timestep = args.resume_step

    model.train(models_dir, save_timesteps=args.save_timesteps, start_timestep=start_timestep)

    env.close()

if __name__ == '__main__':
    main()