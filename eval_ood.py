"""
Out-of-distribution eval for the BC actor on PandaPickAndPlace-v3.

Standalone -- does not touch B_cloning.py / train_bc.py / eval_panda.py.

WHY: the expert demos (scripted_expert.py record()) and the eval loops both use
panda-gym defaults, so the reported success rate is measured on the same
object/goal distribution the demos came from. panda-gym samples object and goal
xy uniformly in a square of +-0.15 m about the table centre
(panda_gym/envs/tasks/pick_and_place.py: obj_xy_range=0.3, goal_xy_range=0.3),
so any position with radius <= 0.15 is inside the training support and anything
beyond 0.15*sqrt(2)=0.212 is outside it entirely.

HOW: PandaPickAndPlaceEnv never forwards those range kwargs to the task
(panda_tasks.py:103 builds `PickAndPlace(sim, reward_type=reward_type)`), so
they cannot be set via gym.make. But `_sample_object` / `_sample_goal` read the
range attributes off `self` at every reset, so replacing the two bound methods on
the task instance is enough -- no subclassing, no registry surgery.

Positions are drawn from an annulus [r_lo, r_hi] about the table centre so success
rate can be plotted against distance-from-centre. Samples that would put the cube
off the table are rejected (the table spans x in [-0.85, 0.25], y in [-0.35, 0.35]).

The scripted expert is run on the identical seeds as a control: it uses privileged
state, so where IT fails the bin is physically infeasible rather than a policy
generalisation failure.

    python eval_ood.py --sanity                 # check sampled positions are valid
    python eval_ood.py --episodes 60            # full sweep + table + plot
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch

from B_cloning import ActorNetwork, RunningMeanStd
from scripted_expert import ScriptedExpert

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENV_ID = "PandaPickAndPlace-v3"

# Table top spans x in [-0.85, 0.25], y in [-0.35, 0.35] (pybullet.py create_table:
# half_extents=[length, width, height]/2 at x_offset=-0.3). Keep the cube centre a
# margin inside that so it cannot topple off an edge.
TABLE_X = (-0.85, 0.25)
TABLE_Y = (-0.35, 0.35)
EDGE_MARGIN = 0.03

# Training support is the +-0.15 square: r<=0.15 is fully inside, r>0.212 fully outside.
BINS = [
    (0.00, 0.05, "in-dist (centre)"),
    (0.05, 0.10, "in-dist"),
    (0.10, 0.15, "in-dist (edge)"),
    (0.15, 0.21, "boundary (corners only)"),
    (0.21, 0.26, "OOD"),
    (0.26, 0.31, "far OOD"),
]


def on_table(xy, half):
    lo_x, hi_x = TABLE_X[0] + EDGE_MARGIN + half, TABLE_X[1] - EDGE_MARGIN - half
    lo_y, hi_y = TABLE_Y[0] + EDGE_MARGIN + half, TABLE_Y[1] - EDGE_MARGIN - half
    return lo_x <= xy[0] <= hi_x and lo_y <= xy[1] <= hi_y


def sample_xy(np_random, r_lo, r_hi, half):
    """Uniform-in-radius point in the annulus, rejecting anything off the table."""
    for _ in range(200):
        r = np_random.uniform(r_lo, r_hi)
        theta = np_random.uniform(0.0, 2.0 * np.pi)
        xy = np.array([r * np.cos(theta), r * np.sin(theta)])
        if on_table(xy, half):
            return xy, False
    return xy, True  # gave up: caller counts it as a rejection failure


def patch_task_sampling(env, r_lo, r_hi):
    """Replace the task's bound samplers so object+goal come from the annulus.

    Goal z keeps panda-gym's own behaviour (uniform in [0, goal_z_range], forced
    onto the table 30% of the time) so only the xy distance varies across bins.
    """
    task = env.unwrapped.task
    half = task.object_size / 2
    stats = {"gave_up": 0}

    def _sample_object():
        xy, gave_up = sample_xy(task.np_random, r_lo, r_hi, half)
        stats["gave_up"] += int(gave_up)
        return np.array([xy[0], xy[1], half])

    def _sample_goal():
        xy, gave_up = sample_xy(task.np_random, r_lo, r_hi, half)
        stats["gave_up"] += int(gave_up)
        noise_z = task.np_random.uniform(0.0, task.goal_range_high[2])
        if task.np_random.random() < 0.3:
            noise_z = 0.0
        return np.array([xy[0], xy[1], half + noise_z])

    task._sample_object = _sample_object
    task._sample_goal = _sample_goal
    return stats


def load_actor(models_dir, ckpt, obs_dim, act_dim, device):
    actor = ActorNetwork(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(f"{models_dir}/Actor/{ckpt}", map_location=device))
    actor.eval()
    # BC writes "normaliser"; the existing SAC checkpoints sit under "Normalizer".
    # Missing stats must be fatal here -- silently evaluating a normalised policy on
    # raw observations looks like a generalisation failure and would corrupt the sweep.
    candidates = [Path(models_dir) / d / ckpt for d in ("normaliser", "Normalizer")]
    norm_path = next((p for p in candidates if p.is_file()), None)
    if norm_path is None:
        raise FileNotFoundError(
            f"No normaliser stats for checkpoint '{ckpt}' in {[str(p.parent) for p in candidates]}"
        )
    normaliser = RunningMeanStd(obs_dim)
    normaliser.load_state_dict(torch.load(norm_path, map_location="cpu", weights_only=False))
    print(f"[ood] normaliser: {norm_path}")
    return actor, normaliser


def bc_action(actor, normaliser, obs, device):
    # Deterministic: tanh(mean), no sampling. Same normaliser stats as training.
    fused = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
    fused = normaliser.normalise(fused)
    with torch.inference_mode():
        mean, log_sd = actor(torch.as_tensor(fused, device=device).unsqueeze(0))
        return torch.tanh(mean).squeeze(0).cpu().numpy().astype(np.float32)


class ExpertPolicy:
    """Wrapper so run_bin's per-episode reset reaches the expert's state machine.

    Passing the bare `expert.act` bound method silently skips reset() and the
    phase leaks across episodes, which tanks the control.
    """

    def __init__(self):
        self.expert = ScriptedExpert()

    def reset(self):
        self.expert.reset()

    def __call__(self, obs):
        return self.expert.act(obs)


def run_bin(env, r_lo, r_hi, episodes, seed0, policy):
    """Roll out `episodes` episodes with annulus sampling. policy(obs) -> action."""
    stats = patch_task_sampling(env, r_lo, r_hi)
    successes, radii = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        radii.append(float(np.linalg.norm(obs["achieved_goal"][:2])))
        done, ok = False, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            ok = ok or bool(info.get("is_success", False))
            done = bool(terminated or truncated)
        successes.append(float(ok))
    return float(np.mean(successes)), float(np.mean(radii)), stats["gave_up"]


def sanity(episodes=6):
    """Check sampled positions land on the table and the cube doesn't fall off."""
    env = gym.make(ENV_ID)
    print("Sanity check: sampled positions per bin (object xy radius, object z after settle)\n")
    for r_lo, r_hi, label in BINS:
        patch_task_sampling(env, r_lo, r_hi)
        rows = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=50_000 + ep)
            obj, goal = obs["achieved_goal"], obs["desired_goal"]
            rows.append((np.linalg.norm(obj[:2]), obj[0], obj[1], obj[2], np.linalg.norm(goal[:2])))
        r = np.array([x[0] for x in rows])
        z = np.array([x[3] for x in rows])
        x = np.array([x[1] for x in rows])
        y = np.array([x[2] for x in rows])
        print(f"[{r_lo:.2f},{r_hi:.2f}] {label:<24} r={r.min():.3f}-{r.max():.3f} "
              f"x={x.min():+.3f}..{x.max():+.3f} y={y.min():+.3f}..{y.max():+.3f} "
              f"z={z.min():.3f}-{z.max():.3f} {'OK' if np.all(np.abs(z - 0.02) < 0.01) else 'CUBE FELL'}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=str, default=f"models/BC_{ENV_ID}")
    parser.add_argument("--checkpoint", type=str, default="95_best")
    parser.add_argument("--episodes", type=int, default=60, help="Episodes per bin (>=50 recommended)")
    parser.add_argument("--seed", type=int, default=20_000, help="Base seed, kept clear of the demo seeds 0-199")
    parser.add_argument("--no-expert", action="store_true", help="Skip the scripted-expert feasibility control")
    parser.add_argument("--sanity", action="store_true", help="Only print sampled positions per bin and exit")
    parser.add_argument("--plot", type=str, default="bc_ood_curve.png")
    args = parser.parse_args()

    if args.sanity:
        sanity()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(ENV_ID)
    obs_dim = (env.observation_space["observation"].shape[0]
               + env.observation_space["desired_goal"].shape[0])
    act_dim = env.action_space.shape[0]
    actor, normaliser = load_actor(args.models_dir, args.checkpoint, obs_dim, act_dim, device)
    print(f"[ood] {args.models_dir}/Actor/{args.checkpoint} | {args.episodes} episodes/bin | device={device}\n")

    expert = ExpertPolicy()
    rows = []
    for r_lo, r_hi, label in BINS:
        bc_rate, mean_r, gave_up = run_bin(
            env, r_lo, r_hi, args.episodes, args.seed,
            lambda obs: bc_action(actor, normaliser, obs, device),
        )
        if args.no_expert:
            exp_rate = float("nan")
        else:
            exp_rate, _, _ = run_bin(env, r_lo, r_hi, args.episodes, args.seed, expert)
        rows.append((r_lo, r_hi, label, mean_r, bc_rate, exp_rate, gave_up))
        print(f"  [{r_lo:.2f},{r_hi:.2f}] {label:<24} BC={bc_rate:6.1%}  expert={exp_rate:6.1%}")

    print(f"\n{'radius bin':<14}{'label':<26}{'mean r':>8}{'BC':>9}{'expert':>9}{'BC/expert':>11}")
    print("-" * 77)
    for r_lo, r_hi, label, mean_r, bc_rate, exp_rate, gave_up in rows:
        ratio = bc_rate / exp_rate if exp_rate > 0 else float("nan")
        print(f"{f'[{r_lo:.2f},{r_hi:.2f}]':<14}{label:<26}{mean_r:>8.3f}"
              f"{bc_rate:>9.1%}{exp_rate:>9.1%}{ratio:>11.2f}")
    env.close()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centres = [(r_lo + r_hi) / 2 for r_lo, r_hi, *_ in rows]
        bc = [r[4] for r in rows]
        exp = [r[5] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(centres, bc, "o-", label="BC policy")
        if not args.no_expert:
            ax.plot(centres, exp, "s--", color="grey", label="scripted expert (feasibility ceiling)")
        ax.axvline(0.15, color="tab:green", ls=":", label="edge of training support (r=0.15)")
        ax.axvline(0.2121, color="tab:red", ls=":", label="fully outside support (r=0.212)")
        ax.set_xlabel("distance of object/goal from table centre (m)")
        ax.set_ylabel("success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"BC generalisation vs distance ({args.episodes} episodes/bin)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\n[ood] wrote plot: {args.plot}")
    except Exception as exc:
        print(f"\n[ood] plot skipped ({exc})")


if __name__ == "__main__":
    main()
