"""
Train SAC+HER on panda-gym's PandaPickAndPlace-v3.

Parallel to train_stack.py (which trains the custom MuJoCo stacking env and is
left untouched). This script builds the panda-gym vector env, a dedicated
reference env for HER reward relabeling, and an rgb_array eval env for demo
frame capture.

    pip install panda-gym
    python train_panda.py --num-envs 8 --timesteps 200000
"""
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  (registers PandaPickAndPlace-v3)
import torch

from SAC_agent_HER_panda import SACAgent

ENV_ID = "PandaPickAndPlace-v3"

# Windows consoles default to cp1252; the training log uses box-drawing chars.
# Force UTF-8 stdout so redirecting to a log file doesn't crash on encode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_env():
    def _init():
        return gym.make(ENV_ID)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-step", type=int, default=None, help="Load actor/critic checkpoint from this timestep")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Number of training steps to run")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--updates-per-step", type=int, default=2, help="Gradient updates per env step")
    parser.add_argument("--debug-boundary", action="store_true", help="Print one-shot env-boundary shape/dtype debug on the first step")
    args = parser.parse_args()

    models_dir = "models/SAC_panda"
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stand up an rgb_array eval env so frames can be captured for a demo clip. ---
    # (Created/verified here; eval_panda.py does the actual rollout + capture.)
    eval_env = gym.make(ENV_ID, render_mode="rgb_array")
    eval_env.reset(seed=0)
    frame = eval_env.render()
    print(f"[eval] rgb_array render OK: frame shape={None if frame is None else np.asarray(frame).shape}")
    eval_env.close()

    # --- Dedicated un-vectorized reference env for HER compute_reward (decision: reference env). ---
    reward_env = gym.make(ENV_ID)

    # --- Vectorized training envs (same wrapper style as train_stack.py). ---
    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(args.num_envs)])

    model = SACAgent(
        env,
        device=device,
        reward_env=reward_env,
        timesteps=args.timesteps,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
    )

    start_timestep = 0
    if args.resume_step is not None:
        model.load_checkpoint(models_dir, args.resume_step)
        start_timestep = args.resume_step

    model.train(
        models_dir,
        save_timesteps=args.save_timesteps,
        start_timestep=start_timestep,
        debug_boundary=args.debug_boundary,
    )

    env.close()
    reward_env.close()


if __name__ == "__main__":
    main()
