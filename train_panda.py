# python train_panda.py --env PandaReach-v3 --warmup 1000 --target-entropy -4 --timesteps 50000
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import panda_gym 
import torch

from SAC_agent_HER_panda import SACAgent

# Windows consoles default to cp1252; the training log uses box-drawing chars.
# Force UTF-8 stdout so redirecting to a log file doesn't crash on encode.
# claude fix :)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def make_env(env_id):
    def _init():
        return gym.make(env_id)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PandaPickAndPlace-v3",
                        help="panda-gym goal env id (e.g. PandaReach-v3, PandaPush-v3, PandaPickAndPlace-v3)")
    parser.add_argument("--resume-step", type=int, default=None, help="Load actor/critic checkpoint from this timestep")
    # NOTE: --timesteps counts GLOBAL steps; each global step advances all envs, so
    # env_steps = timesteps * num_envs. 200k global x 8 envs = 1.6M env-steps
    parser.add_argument("--timesteps", type=int, default=200_000, help="Number of GLOBAL training steps (env_steps = timesteps * num_envs)")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval (global steps)")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--updates-per-step", type=int, default=8, help="Gradient updates per global step (8 = ~1:1 with 8 envs; affordable now the sampler is vectorized)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Actor/critic learning rate (SAC default 3e-4; was 1e-4)")
    parser.add_argument("--warmup", type=int, default=15_000, help="Env-steps of uniform-random actions before using the actor")
    parser.add_argument("--her-ratio", type=float, default=0.8, help="Fraction of each batch that is HER future-relabeled (0.8 = standard; lower = more real-goal transitions)")
    parser.add_argument("--models-dir", type=str, default=None, help="Checkpoint dir (default: models/SAC_<env>)")
    parser.add_argument("--debug-boundary", action="store_true", help="Print one-shot env-boundary shape/dtype debug on the first step")
    args = parser.parse_args()

    # Per-env model dir so Reach/Push/PickAndPlace checkpoints don't collide.
    models_dir = args.models_dir or f"models/SAC_{args.env}"
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_env = gym.make(args.env, render_mode="rgb_array")
    eval_env.reset(seed=0)
    frame = eval_env.render()
    print(f"[eval] {args.env} rgb_array render OK: frame shape={None if frame is None else np.asarray(frame).shape}")
    eval_env.close()

    reward_env = gym.make(args.env)
    env = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])

    model = SACAgent(
        env,
        device=device,
        reward_env=reward_env,
        timesteps=args.timesteps,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        random_explore_steps=args.warmup,
        her_ratio=args.her_ratio,
        lr=args.lr,
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
