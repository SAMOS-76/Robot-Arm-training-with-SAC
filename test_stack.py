import argparse
import re
import time
from pathlib import Path
from typing import Optional

import mujoco.viewer
import numpy as np
import torch
from torch.distributions import Normal

from SAC_agent_HER import ActorNetwork
from stack_block_env import S0100Env


def _extract_step(checkpoint_name: str):
    match = re.match(r"^(\d+)", checkpoint_name)
    if match is None:
        return None
    return int(match.group(1))


def _list_actor_checkpoints(actor_dir: Path):
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"Actor directory not found: {actor_dir}")

    checkpoints = []
    for child in actor_dir.iterdir():
        if not child.is_file():
            continue
        step = _extract_step(child.name)
        if step is None:
            continue
        checkpoints.append((step, child.name))

    if not checkpoints:
        raise FileNotFoundError(f"No actor checkpoints found in: {actor_dir}")

    checkpoints.sort(key=lambda item: (item[0], item[1]))
    return checkpoints


def _resolve_checkpoint_name(actor_dir: Path, checkpoint_arg: Optional[str]):
    checkpoints = _list_actor_checkpoints(actor_dir)

    if checkpoint_arg is None:
        return checkpoints[-1][1]

    direct_path = actor_dir / checkpoint_arg
    if direct_path.is_file():
        return checkpoint_arg

    if checkpoint_arg.isdigit():
        desired_step = int(checkpoint_arg)
        matches = [name for step, name in checkpoints if step == desired_step]
        if not matches:
            raise FileNotFoundError(
                f"No actor checkpoint found for step {desired_step} in {actor_dir}"
            )
        if checkpoint_arg in matches:
            return checkpoint_arg
        matches.sort()
        return matches[-1]

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_arg}' was not found in {actor_dir}. "
        "Use an exact filename or a numeric step."
    )


def _policy_action(actor: ActorNetwork, joints_obs: np.ndarray, device: torch.device, deterministic: bool):
    obs_tensor = torch.as_tensor(joints_obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, log_sd = actor(obs_tensor)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_sd.exp()
            dist = Normal(mean, std)
            action = torch.tanh(dist.sample())

    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _extract_joints(obs):
    if isinstance(obs, dict):
        return obs["joints"]
    return obs


def main():
    parser = argparse.ArgumentParser(description="Visual SAC checkpoint evaluation for stack task")
    parser.add_argument("--models-dir", type=str, default="models/SAC_stacking", help="Model directory containing Actor/Critic/Alpha folders")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename or numeric step. Default: latest in Actor/")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--episode-steps", type=int, default=400, help="Environment episode horizon")
    parser.add_argument("--success-hold-steps", type=int, default=1, help="Consecutive success steps required")
    parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions instead of deterministic tanh(mean)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra sleep per environment step in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    actor_dir = models_dir / "Actor"
    checkpoint_name = _resolve_checkpoint_name(actor_dir, args.checkpoint)
    checkpoint_step = _extract_step(checkpoint_name)

    env = S0100Env(
        render_mode=None,
        max_episode_steps=args.episode_steps,
        success_hold_steps=args.success_hold_steps,
        obs_type="blind",
    )

    obs_dim = env.observation_space["joints"].shape[0]
    act_dim = env.action_space.shape[0]
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim).to(device)

    actor_path = actor_dir / checkpoint_name
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    print(f"Loaded actor checkpoint: {actor_path}")
    if checkpoint_step is not None:
        print(f"Resolved checkpoint step: {checkpoint_step}")
    print(f"Device: {device}")
    print("Launching MuJoCo viewer. Close the viewer window to stop early.")

    episode_returns = []
    episode_successes = []

    try:
        with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
            for episode_idx in range(args.episodes):
                if not viewer.is_running():
                    print("Viewer closed. Stopping evaluation.")
                    break

                episode_seed = None if args.seed is None else args.seed + episode_idx
                obs, _ = env.reset(seed=episode_seed)
                done = False
                episode_return = 0.0
                steps = 0
                last_info = {}

                while not done and viewer.is_running():
                    step_start = time.perf_counter()

                    joints_obs = _extract_joints(obs)
                    action = _policy_action(
                        actor=actor,
                        joints_obs=joints_obs,
                        device=device,
                        deterministic=not args.stochastic,
                    )

                    obs, reward, terminated, truncated, info = env.step(action)
                    viewer.sync()

                    episode_return += float(reward)
                    steps += 1
                    done = bool(terminated or truncated)
                    last_info = info

                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    else:
                        elapsed = time.perf_counter() - step_start
                        delay = max(0.0, float(env.dt) - elapsed)
                        if delay > 0:
                            time.sleep(delay)

                success = bool(last_info.get("success", False))
                episode_returns.append(episode_return)
                episode_successes.append(float(success))

                distances = last_info.get("distances", {}) if isinstance(last_info, dict) else {}
                d_place_red = float(distances.get("dist_place_red", np.nan)) if isinstance(distances, dict) else np.nan
                d_stack_blue = float(distances.get("dist_stack_blue", np.nan)) if isinstance(distances, dict) else np.nan

                print(
                    f"Episode {episode_idx + 1}/{args.episodes} | "
                    f"steps={steps} | return={episode_return:.2f} | success={int(success)} | "
                    f"dist_place_red={d_place_red:.4f} | dist_stack_blue={d_stack_blue:.4f}"
                )
    finally:
        env.close()

    if episode_returns:
        avg_return = float(np.mean(episode_returns))
        success_rate = float(np.mean(episode_successes))
        print("-" * 80)
        print(
            f"Evaluation complete | episodes={len(episode_returns)} | "
            f"avg_return={avg_return:.2f} | success_rate={success_rate:.2%}"
        )
    else:
        print("No episodes were completed.")


if __name__ == "__main__":
    main()
