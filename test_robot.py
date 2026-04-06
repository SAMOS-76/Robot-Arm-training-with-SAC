import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import mujoco

from SAC_agent import ActorNetwork, CNNEncoder
from stack_block_env import S0100Env

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import mujoco.viewer as mj_viewer
except ImportError:
    mj_viewer = None


def find_latest_checkpoint(models_dir: str) -> str:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {models_dir}")

    candidates = []
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isfile(path) and name.isdigit():
            candidates.append((int(name), path))

    if not candidates:
        raise FileNotFoundError(
            f"No numeric checkpoint files found in {models_dir}."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_actor_checkpoint(actor: ActorNetwork, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    actor.load_state_dict(checkpoint)
    actor.eval()


def build_fused_obs(
    obs: dict,
    device: torch.device,
    latent_dim: int,
    encoder: Optional[CNNEncoder] = None,
) -> torch.Tensor:
    joints = torch.as_tensor(obs["joints"], dtype=torch.float32, device=device).unsqueeze(0)

    if encoder is None:
        latent = torch.zeros((1, latent_dim), dtype=torch.float32, device=device)
    else:
        img = torch.as_tensor(obs["image"], dtype=torch.float32, device=device).unsqueeze(0)
        img = (img / 255.0).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            latent = encoder(img)

    return torch.cat([latent, joints], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and view a trained SAC robot-arm policy.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Actor checkpoint path. If omitted, newest checkpoint in models/SAC is used.")
    parser.add_argument("--models-dir", type=str, default="models/SAC_stacking/Actor", help="Directory containing numeric checkpoint files.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=250, help="Max steps per evaluation episode.")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Use tanh(mean) action instead of sampling.")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Sample actions from the actor distribution for stress testing.")
    parser.add_argument("--encoder", type=str, default=None, help="Optional encoder checkpoint path for vision-based evaluation.")
    parser.add_argument("--viewer", choices=["mujoco", "frames", "none"], default="mujoco", help="Viewer backend: external MuJoCo window, image-frame window, or no viewer.")
    parser.add_argument("--render-sleep", type=float, default=0.02, help="Sleep (seconds) after each viewer update to slow playback for visibility.")
    parser.add_argument("--no-view", action="store_true", help="Disable live frame viewer.")
    parser.set_defaults(deterministic=True)
    return parser.parse_args()


def create_viewer(mode: str, env: S0100Env):
    if mode == "none":
        return {"type": "none"}

    if mode == "mujoco":
        if mj_viewer is None:
            print("MuJoCo viewer module unavailable, falling back to frame viewer.")
            mode = "frames"
        else:
            viewer_ctx = mj_viewer.launch_passive(env.model, env.data)
            viewer = viewer_ctx.__enter__()
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.array([0.22, 0.0, 0.14], dtype=np.float64)
            viewer.cam.distance = 0.75
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -22.0
            return {"type": "mujoco", "viewer": viewer, "ctx": viewer_ctx}

    if cv2 is not None:
        cv2.namedWindow("SAC Evaluation", cv2.WINDOW_NORMAL)
        return {"type": "cv2"}

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    artist = ax.imshow(np.zeros((84, 84, 3), dtype=np.uint8))
    ax.set_title("SAC Evaluation")
    ax.axis("off")
    return {"type": "mpl", "fig": fig, "artist": artist}


def update_viewer(viewer: dict, frame: Optional[np.ndarray], sleep_s: float = 0.0) -> bool:
    viewer_type = viewer.get("type", "none")
    if viewer_type == "none":
        return True

    if viewer_type == "mujoco":
        mj = viewer["viewer"]
        if not mj.is_running():
            return False
        mj.sync()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return True

    if viewer_type == "cv2":
        if frame is None:
            return True
        cv2.imshow("SAC Evaluation", frame[:, :, ::-1])
        cv2.waitKey(1)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return True

    if viewer_type == "mpl":
        if frame is None:
            return True
        viewer["artist"].set_data(frame)
        viewer["fig"].canvas.draw_idle()
        plt.pause(max(0.001, sleep_s))

    return True


def close_viewer(viewer: dict) -> None:
    viewer_type = viewer.get("type", "none")
    if viewer_type == "mujoco":
        viewer["ctx"].__exit__(None, None, None)
    elif viewer_type == "cv2":
        cv2.destroyAllWindows()
    elif viewer_type == "mpl":
        plt.ioff()
        plt.close(viewer["fig"])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = S0100Env(render_mode=None, max_episode_steps=args.max_steps)
    viewer_mode = "none" if args.no_view else args.viewer
    viewer = create_viewer(mode=viewer_mode, env=env)
    joint_dim = env.observation_space["joints"].shape[0]
    act_dim = env.action_space.shape[0]
    latent_dim = 50
    fused_obs_dim = latent_dim + joint_dim

    actor = ActorNetwork(fused_obs_dim, act_dim).to(device)
    checkpoint_path = args.checkpoint or find_latest_checkpoint(args.models_dir)
    load_actor_checkpoint(actor, checkpoint_path, device)
    print(f"Loaded actor checkpoint: {checkpoint_path}")

    encoder = None
    if args.encoder is not None:
        image_shape = env.observation_space["image"].shape
        encoder = CNNEncoder(image_shape=image_shape, latent_dim=latent_dim).to(device)
        enc_state = torch.load(args.encoder, map_location=device)
        if isinstance(enc_state, dict) and "state_dict" in enc_state:
            enc_state = enc_state["state_dict"]
        encoder.load_state_dict(enc_state)
        encoder.eval()
        print(f"Loaded encoder checkpoint: {args.encoder}")
    else:
        print("No encoder checkpoint provided. Using zero latent vector fallback.")

    returns = []
    successes = []
    final_distances = []

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            running = update_viewer(viewer, obs.get("image"), sleep_s=args.render_sleep)
            if not running:
                print("MuJoCo viewer closed by user. Stopping evaluation.")
                break

            terminated = False
            truncated = False
            ep_return = 0.0
            ep_steps = 0
            last_info = {}

            while not (terminated or truncated):
                with torch.no_grad():
                    fused_obs = build_fused_obs(obs, device, latent_dim, encoder)
                    mean, log_sd = actor(fused_obs)

                    if args.deterministic:
                        action = torch.tanh(mean)
                    else:
                        sd = log_sd.exp()
                        noise = torch.randn_like(mean)
                        action = torch.tanh(mean + sd * noise)

                action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action_np)
                running = update_viewer(viewer, obs.get("image"), sleep_s=args.render_sleep)
                if not running:
                    print("MuJoCo viewer closed by user. Stopping evaluation.")
                    truncated = True
                    break

                ep_return += float(reward)
                ep_steps += 1
                last_info = info

            ep_success = bool(last_info.get("success", False))
            ep_dist = float(last_info.get("distance", np.nan))

            returns.append(ep_return)
            successes.append(1.0 if ep_success else 0.0)
            final_distances.append(ep_dist)

            print(
                f"Episode {ep}/{args.episodes} | steps={ep_steps} | "
                f"return={ep_return:.3f} | success={ep_success} | final_distance={ep_dist:.4f}"
            )

        print("\nEvaluation summary")
        print(f"Mean return: {np.mean(returns):.3f}")
        print(f"Success rate: {np.mean(successes):.3f}")
        print(f"Mean final distance: {np.nanmean(final_distances):.4f}")
    finally:
        close_viewer(viewer)
        env.close()


if __name__ == "__main__":
    main()