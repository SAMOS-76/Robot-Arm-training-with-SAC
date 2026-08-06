"""
Roll out an actor checkpoint on PandaPickAndPlace-v3 and write a GIF.

Prints the success rate over the rollout episodes, so the same command doubles as a
quick eyeball check and a way to produce a demo clip. Works for SAC and BC checkpoints
alike, they share the actor architecture.

    python eval_panda.py --checkpoint 50000 --episodes 5
"""
import argparse
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
from torch.distributions import Normal

from SAC_agent_HER_panda import ActorNetwork, RunningMeanStd


def policy_action(actor, obs, device, normaliser=None, deterministic=True):
    # Policy input = concat(observation, desired_goal); achieved_goal excluded
    obs_input = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    if normaliser is not None:
        obs_input = normaliser.normalise(obs_input[None, :])[0]
    obs_tensor = torch.as_tensor(obs_input, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, log_sd = actor(obs_tensor)
        action = torch.tanh(mean) if deterministic else torch.tanh(Normal(mean, log_sd.exp()).sample())
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def resolve_checkpoint(actor_dir, name):
    # Named checkpoint if it exists, otherwise the highest numeric one
    if name and (actor_dir / name).is_file():
        return actor_dir / name
    numbered = [(int(p.name.split("_")[0]), p) for p in sorted(actor_dir.glob("*"))
                if p.is_file() and p.name.split("_")[0].isdigit()]
    if not numbered:
        raise FileNotFoundError(f"No actor checkpoints in {actor_dir}")
    if name:
        match = [p for step, p in numbered if str(step) == name]
        if match:
            return match[-1]
    return max(numbered, key=lambda x: x[0])[1]


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
    parser.add_argument("--env", type=str, default="PandaPickAndPlace-v3",
                        help="panda-gym goal env id (match the one used for training)")
    parser.add_argument("--models-dir", type=str, default=None, help="Checkpoint dir (default: models/SAC_<env>)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or numeric step in Actor/. Default: latest")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--out", type=str, default="panda_eval.gif")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    models_dir = Path(args.models_dir or f"models/SAC_{args.env}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(models_dir / "Actor", args.checkpoint)

    env = gym.make(args.env, render_mode="rgb_array")
    obs_dim = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
    act_dim = env.action_space.shape[0]
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim).to(device)
    actor.load_state_dict(torch.load(ckpt, map_location=device))
    actor.eval()
    print(f"[eval] loaded {ckpt} | obs_dim={obs_dim} act_dim={act_dim} | device={device}")

    # Training normalised the network input, so the matching stats have to come along.
    # Older checkpoints used "Normalizer", the agent now writes "normaliser".
    norm_paths = [models_dir / d / ckpt.name for d in ("normaliser", "Normalizer")]
    norm_path = next((p for p in norm_paths if p.is_file()), None)
    normaliser = None
    if norm_path is not None:
        normaliser = RunningMeanStd(obs_dim)
        normaliser.load_state_dict(torch.load(norm_path, map_location="cpu", weights_only=False))
        print(f"[eval] loaded normaliser stats: {norm_path}")
    else:
        print(f"[eval] WARNING: no normaliser in {[str(p.parent) for p in norm_paths]} — feeding UNNORMALISED obs (policy may look broken)")

    frames, successes = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_success = False
        while not done:
            frames.append(np.asarray(env.render()))
            action = policy_action(actor, obs, device, normaliser=normaliser, deterministic=not args.stochastic)
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
