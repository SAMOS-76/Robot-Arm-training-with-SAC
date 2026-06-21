"""
Evaluate / capture a SAC actor checkpoint on PandaPickAndPlace-v3.

Rolls out the policy on an rgb_array env and writes a GIF (and prints success
rate) so you can produce a demo clip. Replaces test_stack.py, which used the
MuJoCo viewer specific to the custom stacking env.

    python eval_panda.py --checkpoint 50000 --episodes 5
"""
import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch

from SAC_agent_HER_panda import ActorNetwork

ENV_ID = "PandaPickAndPlace-v3"


def policy_action(actor, obs, device, deterministic=True):
    # [PANDA] policy input = concat(observation, desired_goal); achieved_goal excluded.
    obs_input = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, log_sd = actor(obs_tensor)
        if deterministic:
            action = torch.tanh(mean)
        else:
            from torch.distributions import Normal
            action = torch.tanh(Normal(mean, log_sd.exp()).sample())
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def save_gif(frames, path, fps=25):
    if not frames:
        print("[eval] no frames captured")
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"[eval] wrote GIF: {path} ({len(frames)} frames)")
    except Exception as exc:
        # Fallback: dump frames as PNGs via matplotlib (already a project dep).
        import matplotlib.image as mpimg
        out_dir = Path(path).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            mpimg.imsave(out_dir / f"frame_{i:04d}.png", fr)
        print(f"[eval] imageio unavailable ({exc}); wrote {len(frames)} PNGs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default="models/SAC_panda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or numeric step in Actor/. Default: latest")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--out", type=str, default="panda_eval.gif")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_dir = Path(args.models_dir) / "Actor"

    # Resolve checkpoint (latest numeric if unspecified).
    if args.checkpoint and (actor_dir / args.checkpoint).is_file():
        ckpt = actor_dir / args.checkpoint
    else:
        cands = []
        for c in actor_dir.iterdir() if actor_dir.is_dir() else []:
            head = c.name.split("_")[0]
            if c.is_file() and head.isdigit():
                cands.append((int(head), c))
        if not cands:
            raise FileNotFoundError(f"No actor checkpoints in {actor_dir}")
        if args.checkpoint:
            match = [c for s, c in cands if str(s) == args.checkpoint]
            ckpt = match[-1] if match else max(cands, key=lambda x: x[0])[1]
        else:
            ckpt = max(cands, key=lambda x: x[0])[1]

    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs_dim = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim).to(device)
    actor.load_state_dict(torch.load(ckpt, map_location=device))
    actor.eval()
    print(f"[eval] loaded {ckpt} | obs_dim={obs_dim} act_dim={act_dim} | device={device}")

    frames = []
    successes = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_success = False
        while not done:
            frames.append(np.asarray(env.render()))
            action = policy_action(actor, obs, device, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_success = ep_success or bool(info.get("is_success", False))
            done = bool(terminated or truncated)
        successes.append(float(ep_success))
        print(f"[eval] episode {ep + 1}/{args.episodes} success={int(ep_success)}")

    env.close()
    print(f"[eval] success_rate={np.mean(successes):.2%} over {len(successes)} episodes")
    save_gif(frames, args.out)


if __name__ == "__main__":
    main()
