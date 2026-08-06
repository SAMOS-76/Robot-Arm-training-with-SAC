"""
Claude Scripted expert for panda-gym's `PandaPickAndPlace-v3`.

Uses privileged state (object / gripper / goal positions from the obs dict) to
drive a state-machine + proportional controller. Generates demonstrations for
behavior cloning + SAC/HER fine-tuning.

The obs dict (see .venv/.../panda_gym/envs/core.py:_get_obs) has keys:
    observation   (19,)  ->  ee_pos[0:3], ee_vel[3:6], fingers_width[6], object...[7:]
    achieved_goal (3,)   ->  object position
    desired_goal  (3,)   ->  target position
Action (4,), each in [-1, 1]: ee_disp = action[:3] * 0.05 m/step; gripper
action[3] is POSITIVE=open, NEGATIVE=close (panda.py:set_action). So a
proportional gain of ~1/0.05 = 20 saturates the displacement in one step.

The env terminates the instant the object is within 0.05 m of the goal
(core.py: terminated = is_success) and TimeLimit-truncates at 50 steps. Because
success fires while the object is still grasped, the expert HOLDS the grasp all
the way to the goal and never releases (the RELEASE branch is unreached on
success, kept only for completeness).

Usage:
    python scripted_expert.py --verify --episodes 10        # eyeball first
    python scripted_expert.py --record --episodes 200 --out expert_demos.npz

NOTE on `dones`: the recorded dataset stores the real env terminal flag
(terminated or truncated). SAC_agent_HER_panda.py's replay buffer overrides
stored done to 0 (panda episodes are treated as non-terminal for bootstrapping),
so this field is informational for BC and does not conflict with the HER loader.
"""
import argparse
import sys
import time

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  (registers PandaPickAndPlace-v3)

# Windows consoles default to cp1252; force UTF-8 so redirecting logs doesn't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENV_ID = "PandaPickAndPlace-v3"

# --- Controller tuning constants (meters) -----------------------------------
GAIN = 20.0            # proportional gain: ee_disp = action*0.05, so 20 saturates in 1 step
HOVER_HEIGHT = 0.10    # height above the object to hover at before descending
LIFT_HEIGHT = 0.20     # absolute world z to lift the grasped object to
XY_ALIGN_TOL = 0.02    # xy error below which we start descending
GRASP_TOL = 0.015      # ee-to-object distance below which we close the gripper
LIFT_TOL = 0.03        # how close the object z must get to LIFT_HEIGHT to count as "lifted"
GRASP_SETTLE_STEPS = 4 # steps to hold while the gripper closes before lifting
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# Phases
ABOVE, DESCEND, GRASP, LIFT, TO_GOAL, RELEASE = range(6)


class ScriptedExpert:
    """Per-episode state-machine controller. Call reset() then act(obs) each step."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = ABOVE
        self.grasp_counter = 0

    @staticmethod
    def _servo(target, current, gripper):
        """Proportional action toward `target` xyz with a given gripper command."""
        disp = GAIN * (np.asarray(target) - np.asarray(current))
        action = np.concatenate([disp, [gripper]]).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def act(self, obs):
        ee_pos = obs["observation"][0:3]
        object_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]

        above_object = np.array([object_pos[0], object_pos[1], object_pos[2] + HOVER_HEIGHT])

        if self.phase == ABOVE:
            # Align in xy above the object, keep gripper open.
            xy_err = np.linalg.norm(ee_pos[:2] - object_pos[:2])
            if xy_err < XY_ALIGN_TOL:
                self.phase = DESCEND
                return self.act(obs)
            return self._servo(above_object, ee_pos, GRIPPER_OPEN)

        if self.phase == DESCEND:
            # Lower onto the object, gripper still open.
            if np.linalg.norm(ee_pos - object_pos) < GRASP_TOL:
                self.phase = GRASP
                self.grasp_counter = 0
                return self.act(obs)
            return self._servo(object_pos, ee_pos, GRIPPER_OPEN)

        if self.phase == GRASP:
            # Hold position and close the gripper for a few steps.
            self.grasp_counter += 1
            if self.grasp_counter >= GRASP_SETTLE_STEPS:
                self.phase = LIFT
            return self._servo(object_pos, ee_pos, GRIPPER_CLOSE)

        if self.phase == LIFT:
            # Raise the grasped object to a fixed absolute height, gripper closed.
            # Target must be absolute: a target relative to the (rising) object
            # would recede upward and never be reached.
            if object_pos[2] >= LIFT_HEIGHT - LIFT_TOL:
                self.phase = TO_GOAL
                return self.act(obs)
            lift_target = np.array([object_pos[0], object_pos[1], LIFT_HEIGHT])
            return self._servo(lift_target, ee_pos, GRIPPER_CLOSE)

        if self.phase == TO_GOAL:
            # Servo so the object moves to the goal; keep holding.
            # Command the ee to (goal + current ee-object offset) so the grasped
            # object tracks the goal rather than the gripper.
            offset = ee_pos - object_pos
            ee_target = goal_pos + offset
            return self._servo(ee_target, ee_pos, GRIPPER_CLOSE)

        # RELEASE: unreached on success (env terminates while grasped). Open gripper.
        return np.array([0.0, 0.0, 0.0, GRIPPER_OPEN], dtype=np.float32)


def run_episode(env, expert, seed=None, record_traj=False, step_delay=0.0):
    """Roll out one episode. Returns (success, length, transitions, traj).

    transitions: list of per-step dicts with the fields the SAC buffer needs.
    traj: optional per-step object/goal/ee positions for inspection.
    step_delay: seconds to sleep after each step (paces the human-render GUI,
        which otherwise finishes an episode faster than the window can paint).
    """
    obs, info = env.reset(seed=seed)
    expert.reset()
    transitions = []
    traj = {"object_pos": [], "goal_pos": [], "ee_pos": []} if record_traj else None

    success = False
    length = 0
    done = False
    while not done:
        action = expert.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        length += 1
        if step_delay:
            time.sleep(step_delay)

        transitions.append({
            "observation": np.asarray(obs["observation"], dtype=np.float32),
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32),
            "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float32),
            "next_observation": np.asarray(next_obs["observation"], dtype=np.float32),
            "next_achieved_goal": np.asarray(next_obs["achieved_goal"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float32(reward),
            "done": np.float32(done),
        })

        if record_traj:
            traj["object_pos"].append(np.asarray(obs["achieved_goal"], dtype=np.float32))
            traj["goal_pos"].append(np.asarray(obs["desired_goal"], dtype=np.float32))
            traj["ee_pos"].append(np.asarray(obs["observation"][0:3], dtype=np.float32))

        success = success or bool(terminated) or bool(info.get("is_success", False))
        obs = next_obs

    return success, length, transitions, traj


def verify(episodes, render):
    env = gym.make(ENV_ID, render_mode="human") if render else gym.make(ENV_ID)
    expert = ScriptedExpert()
    # Human-mode sim steps run unthrottled (no internal sleep), so an episode
    # finishes and the window closes before it's ever visible. Pace to render_fps.
    step_delay = 1.0 / env.metadata.get("render_fps", 25) if render else 0.0

    successes, lengths = [], []
    saved_trajs = {}
    for ep in range(episodes):
        record_traj = ep < 2  # keep the first two episodes for eyeballing
        success, length, _, traj = run_episode(env, expert, seed=ep, record_traj=record_traj,
                                                step_delay=step_delay)
        successes.append(float(success))
        lengths.append(length)
        print(f"[verify] episode {ep + 1}/{episodes}: success={int(success)} length={length}")
        if record_traj:
            for k, v in traj.items():
                saved_trajs[f"ep{ep}_{k}"] = np.asarray(v, dtype=np.float32)
    env.close()

    rate = float(np.mean(successes))
    print(f"\n[verify] success_rate={rate:.1%} over {episodes} episodes | "
          f"mean_length={np.mean(lengths):.1f}")

    np.savez_compressed("expert_trajectories.npz", **saved_trajs)
    print("[verify] saved first-2-episode trajectories to expert_trajectories.npz "
          "(keys: ep{0,1}_{object_pos,goal_pos,ee_pos})")
    print(f"[verify] success rate = {rate:.1%} — rerun with --record once satisfied.")


def record(episodes, out):
    env = gym.make(ENV_ID)
    expert = ScriptedExpert()

    # Per-transition accumulators (only successful episodes are committed).
    keys = ["observation", "achieved_goal", "desired_goal",
            "next_observation", "next_achieved_goal", "action", "reward", "done"]
    data = {k: [] for k in keys}
    episode_starts = []
    episode_index = []

    kept = 0
    dropped = 0
    successes = []
    for ep in range(episodes):
        success, length, transitions, _ = run_episode(env, expert, seed=ep)
        successes.append(float(success))
        if not success:
            dropped += 1
            continue
        for t, tr in enumerate(transitions):
            for k in keys:
                data[k].append(tr[k])
            episode_starts.append(t == 0)
            episode_index.append(kept)
        kept += 1
        if (ep + 1) % 20 == 0:
            print(f"[record] {ep + 1}/{episodes} attempted | kept={kept} dropped={dropped}")
    env.close()

    if kept == 0:
        print("[record] no successful episodes — nothing saved. Check the controller.")
        return

    arrays = {
        "observation": np.asarray(data["observation"], dtype=np.float32),
        "achieved_goal": np.asarray(data["achieved_goal"], dtype=np.float32),
        "desired_goal": np.asarray(data["desired_goal"], dtype=np.float32),
        "next_observation": np.asarray(data["next_observation"], dtype=np.float32),
        "next_achieved_goal": np.asarray(data["next_achieved_goal"], dtype=np.float32),
        "actions": np.asarray(data["action"], dtype=np.float32),
        "rewards": np.asarray(data["reward"], dtype=np.float32),
        "dones": np.asarray(data["done"], dtype=np.float32),
        "episode_starts": np.asarray(episode_starts, dtype=bool),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
    }
    np.savez_compressed(out, **arrays)

    rate = float(np.mean(successes))
    n_trans = arrays["observation"].shape[0]
    print(f"\n[record] success_rate={rate:.1%} over {episodes} attempted episodes")
    print(f"[record] kept {kept} successful episodes, dropped {dropped} failures")
    print(f"[record] wrote {n_trans} transitions to {out}")
    print(f"[record] shapes: observation={arrays['observation'].shape} "
          f"action={arrays['actions'].shape} achieved_goal={arrays['achieved_goal'].shape}")


def save_gif(frames, path, fps=25):
    """Write captured rgb_array frames to a GIF (falls back to PNGs if imageio lacks a writer)."""
    if not frames:
        print("[video] no frames captured")
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"[video] wrote {path} ({len(frames)} frames)")
    except Exception as exc:
        import matplotlib.image as mpimg
        from pathlib import Path
        out_dir = Path(path).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            mpimg.imsave(out_dir / f"frame_{i:04d}.png", fr)
        print(f"[video] imageio unavailable ({exc}); wrote {len(frames)} PNGs to {out_dir}")


def capture_video(episodes, out, fps):
    """Re-run the scripted expert with rgb_array rendering and save the rollout as a GIF.

    Uses the same seed=ep scheme as record(), so with the same code these are the
    same episodes that ended up in expert_demos.npz (env + expert are deterministic).
    """
    env = gym.make(ENV_ID, render_mode="rgb_array")
    expert = ScriptedExpert()

    frames = []
    successes = []
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        expert.reset()
        frames.append(np.asarray(env.render()))
        done = False
        ep_success = False
        while not done:
            action = expert.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(np.asarray(env.render()))
            done = bool(terminated or truncated)
            ep_success = ep_success or bool(terminated) or bool(info.get("is_success", False))
        successes.append(float(ep_success))
        print(f"[video] episode {ep + 1}/{episodes}: success={int(ep_success)}")
    env.close()

    print(f"[video] success_rate={np.mean(successes):.1%} over {episodes} episodes")
    save_gif(frames, out, fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true", help="Roll out and print success/length; save inspection trajectories. Does NOT record the dataset.")
    mode.add_argument("--record", action="store_true", help="Record the demonstration dataset (successful episodes only).")
    mode.add_argument("--video", action="store_true", help="Re-run episodes with rgb_array rendering and save a GIF (use this over SSH, where --render's GUI can't display).")
    parser.add_argument("--episodes", type=int, default=None, help="Episodes (default: 10 for --verify/--video, 200 for --record)")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: expert_demos.npz for --record, expert_demo.gif for --video)")
    parser.add_argument("--render", action="store_true", help="Render human viewer during --verify")
    parser.add_argument("--fps", type=int, default=25, help="Playback fps for --video")
    args = parser.parse_args()

    if args.verify:
        verify(episodes=args.episodes or 10, render=args.render)
    elif args.video:
        capture_video(episodes=args.episodes or 10, out=args.out or "expert_demo.gif", fps=args.fps)
    else:
        record(episodes=args.episodes or 200, out=args.out or "expert_demos.npz")


if __name__ == "__main__":
    main()
