# python train_bc.py --epochs 50 --demos expert_demos.npz
import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import panda_gym
import torch

from B_cloning import BC

# Windows consoles default to cp1252, force UTF-8 so redirecting to a log file
# doesn't crash on encode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PandaPickAndPlace-v3",
                        help="panda-gym goal env id (match the one the demos were recorded on)")
    parser.add_argument("--demos", type=str, default="expert_demos.npz", help="Expert demo .npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Small on purpose, ~3k demo transitions means a big batch gives too few updates per epoch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-epochs", type=int, default=5, help="Checkpoint + rollout eval interval (epochs)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Rollout episodes per eval (0 = loss only)")
    parser.add_argument("--models-dir", type=str, default=None, help="Checkpoint dir (default: models/BC_<env>)")
    parser.add_argument("--no-obs-norm", action="store_true", help="Disable observation normalisation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Per-env model dir so BC and SAC checkpoints don't collide.
    models_dir = args.models_dir or f"models/BC_{args.env}"
    os.makedirs(models_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env)

    model = BC(
        env,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        normalise_obs=not args.no_obs_norm,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"BC on {args.env} | obs_dim={model.fused_obs_dim} act_dim={model.act_dim} | device={device}")

    model.train(
        args.demos,
        models_dir,
        save_epochs=args.save_epochs,
        eval_env=env if args.eval_episodes > 0 else None,
        eval_episodes=args.eval_episodes,
    )

    env.close()
    print(f"\nSaved to {models_dir}. Record a GIF with:\n"
          f"  python eval_panda.py --env {args.env} --models-dir {models_dir} --out bc_eval.gif")


if __name__ == "__main__":
    main()
