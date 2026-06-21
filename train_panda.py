"""
Train SAC+HER on a panda-gym goal env (PandaPickAndPlace-v3 by default).

Parallel to train_stack.py (which trains the custom MuJoCo stacking env and is
left untouched). This script builds the panda-gym vector env, a dedicated
reference env for HER reward relabeling, and an rgb_array eval env for demo
frame capture.

    pip install panda-gym

    # Hard task (default): PickAndPlace with exploration-friendly config
    python train_panda.py --num-envs 8

    # Quick pipeline sanity check: Reach solves in ~15k env-steps -> use a small
    # warmup and the standard target entropy so it isn't stuck in warmup.
    python train_panda.py --env PandaReach-v3 --warmup 1000 --target-entropy -4 --timesteps 50000
"""
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  (registers the Panda* envs)
import torch

from SAC_agent_HER_panda import SACAgent

# Windows consoles default to cp1252; the training log uses box-drawing chars.
# Force UTF-8 stdout so redirecting to a log file doesn't crash on encode.
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
    # env_steps = timesteps * num_envs. 200k global x 8 envs = 1.6M env-steps, a sane budget for
    # PickAndPlace (the hard task); the run auto-stops early once 80% success is held.
    parser.add_argument("--timesteps", type=int, default=200_000, help="Number of GLOBAL training steps (env_steps = timesteps * num_envs)")
    parser.add_argument("--save-timesteps", type=int, default=12_500, help="Checkpoint save interval (global steps)")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--replay-size", type=int, default=1_000_000, help="Replay capacity in transitions")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--updates-per-step", type=int, default=4, help="Gradient updates per global step")
    # [RETUNE] exploration knobs (see SAC_agent_HER_panda.py). Defaults target the HARD task
    # (PickAndPlace): a bigger random warmup to discover grasps, and a less-negative target entropy
    # so alpha doesn't collapse to ~0 and kill exploration before the policy ever succeeds.
    parser.add_argument("--warmup", type=int, default=15_000, help="Env-steps of uniform-random actions before using the actor")
    parser.add_argument("--target-entropy", type=float, default=-2.0, help="SAC target entropy for auto-alpha (-act_dim is -4 for panda; less-negative = more exploration)")
    parser.add_argument("--fixed-alpha", type=float, default=None, help="Pin a constant entropy temperature (disables auto-alpha). Try ~0.05 if auto-alpha keeps collapsing to ~0")
    parser.add_argument("--her-ratio", type=float, default=0.8, help="Fraction of each batch that is HER future-relabeled (0.8 = standard; lower = more real-goal transitions)")
    parser.add_argument("--models-dir", type=str, default=None, help="Checkpoint dir (default: models/SAC_<env>)")
    parser.add_argument("--debug-boundary", action="store_true", help="Print one-shot env-boundary shape/dtype debug on the first step")
    args = parser.parse_args()

    # Per-env model dir so Reach/Push/PickAndPlace checkpoints don't collide.
    models_dir = args.models_dir or f"models/SAC_{args.env}"
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Stand up an rgb_array eval env so frames can be captured for a demo clip. ---
    # (Created/verified here; eval_panda.py does the actual rollout + capture.)
    eval_env = gym.make(args.env, render_mode="rgb_array")
    eval_env.reset(seed=0)
    frame = eval_env.render()
    print(f"[eval] {args.env} rgb_array render OK: frame shape={None if frame is None else np.asarray(frame).shape}")
    eval_env.close()

    # --- Dedicated un-vectorized reference env for HER compute_reward (decision: reference env). ---
    reward_env = gym.make(args.env)

    # --- Vectorized training envs (same wrapper style as train_stack.py). ---
    env = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])

    model = SACAgent(
        env,
        device=device,
        reward_env=reward_env,
        timesteps=args.timesteps,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        target_entropy=args.target_entropy,
        random_explore_steps=args.warmup,
        fixed_alpha=args.fixed_alpha,
        her_ratio=args.her_ratio,
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
