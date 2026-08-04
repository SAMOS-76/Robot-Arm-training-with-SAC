"""
Out-of-distribution generalisation comparison on `PandaPickAndPlace-v3`:
behaviour cloning vs SAC+HER, with the scripted expert as a feasibility ceiling.

Standalone and evaluation-only -- does not touch B_cloning.py / train_bc.py /
SAC_agent_HER_panda.py / eval_panda.py / scripted_expert.py.

WHY: panda-gym samples object and goal xy uniformly in a +-0.15 m square about
the table centre (tasks/pick_and_place.py: obj_xy_range=0.3, goal_xy_range=0.3),
and goal z in [0, 0.2] forced flat 30% of the time. Both the expert demos and the
default eval loops draw from exactly that distribution, so the headline success
rate says nothing about what happens when the scene moves outside it.

HOW: PandaPickAndPlaceEnv never forwards the range kwargs to the task
(panda_tasks.py:103 builds `PickAndPlace(sim, reward_type=reward_type)`), so they
cannot be set via gym.make. But `_sample_object` / `_sample_goal` are read off
`self` at every reset, so replacing the two bound methods on the task instance is
enough -- no subclassing, no gym registry surgery.

The samplers installed here return positions from a PRE-GENERATED scene table
rather than drawing fresh randomness. That is the point: every arm replays the
identical scenes, so the comparison is paired and McNemar's exact test applies.
If each policy drew from the env RNG instead, two policies consuming different
numbers of draws would silently desynchronise the scenes.

Geometry: positions are centre + r*(cos t, sin t) with t restricted to the SAME
arc [48.2 deg, 311.8 deg] at every radius. The table ends at x = +0.25, so a full
ring is impossible past r ~ 0.20; rejecting per-radius instead would make the
angular distribution co-vary with radius and confound distance with direction.
Experiment 1 (untouched default sampling) recovers the excluded +x wedge.

    python eval_ood.py --sanity      # position checks, replay check, expert probe
    python eval_ood.py               # all three experiments + csv + plots
"""
import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401  (registers PandaPickAndPlace-v3)
import torch
from scipy.stats import binomtest

from B_cloning import ActorNetwork, RunningMeanStd
from scripted_expert import ScriptedExpert

# Windows consoles default to cp1252; force UTF-8 so redirecting logs doesn't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENV_ID = "PandaPickAndPlace-v3"

CUBE_HALF = 0.02          # object_size 0.04 / 2 -- also the resting cube-centre z
MIN_SEPARATION = 0.05     # == task.distance_threshold; closer than this auto-succeeds at step 1

# Table top spans x in [-0.85, 0.25], y in [-0.35, 0.35] (pybullet.py create_table:
# length=1.1, width=0.7, x_offset=-0.3). Keep the cube centre a 0.03 margin plus its
# own half-extent inside that, so it cannot topple off an edge.
SAFE_X = (-0.85 + 0.03 + CUBE_HALF, 0.25 - 0.03 - CUBE_HALF)   # (-0.80, 0.20)
SAFE_Y = (-0.35 + 0.03 + CUBE_HALF, 0.35 - 0.03 - CUBE_HALF)   # (-0.30, 0.30)

# Fixed safe arc, identical at every radius: cos(t) <= 0.20 / r_max with r_max = 0.30.
# Excludes a 96.4 deg wedge around +x. Beyond r = 0.30 the y-bound also binds and the
# safe region splits into two disconnected arcs, so 0.30 is the cap.
THETA_LO = np.deg2rad(48.2)
THETA_HI = np.deg2rad(311.8)

RADII = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
GOAL_ZS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
HEIGHT_SWEEP_RADIUS = 0.10   # in-distribution xy, so only height is OOD

# panda-gym's own goal-z distribution: 0.02 + U(0, 0.2), flattened to 0.02 w.p. 0.3.
GOAL_Z_RANGE = 0.2
GOAL_Z_FLAT_PROB = 0.3
TRAIN_Z_CEILING = CUBE_HALF + GOAL_Z_RANGE   # 0.22 absolute

SCENE_RNG_SEED = 12345
BASE_SEED = 20_000        # clear of the demo seeds 0-199 (scripted_expert.py record())

ARMS = ["BC", "SAC+HER", "expert"]
ARM_COLOUR = {"BC": "#2a78d6", "SAC+HER": "#eb6834", "expert": "#1baf7a"}
ARM_MARKER = {"BC": "o", "SAC+HER": "s", "expert": "^"}

# Chart chrome (light surface).
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


# --------------------------------------------------------------------------- #
# scene table
# --------------------------------------------------------------------------- #
def in_safe_rect(xy):
    return (SAFE_X[0] <= xy[0] <= SAFE_X[1]) and (SAFE_Y[0] <= xy[1] <= SAFE_Y[1])


def _ring_xy(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def _default_goal_z(rng):
    """panda-gym's goal-z distribution, drawn from our own rng."""
    z = rng.uniform(0.0, GOAL_Z_RANGE)
    if rng.random() < GOAL_Z_FLAT_PROB:
        z = 0.0
    return CUBE_HALF + z


def generate_ring_scenes(rng, n, radius, goal_z=None):
    """n scenes with object and goal on a ring of `radius` at independent safe-arc angles.

    goal_z None -> panda-gym's default height distribution (radius sweep).
    goal_z float -> that absolute height for every scene (height sweep).
    Scenes with initial object-goal separation < 0.05 m are rejected and resampled.
    """
    objects, goals, thetas, rejected = [], [], [], 0
    while len(objects) < n:
        th_o = rng.uniform(THETA_LO, THETA_HI)
        th_g = rng.uniform(THETA_LO, THETA_HI)
        z = _default_goal_z(rng) if goal_z is None else float(goal_z)
        obj = np.append(_ring_xy(radius, th_o), CUBE_HALF)
        goal = np.append(_ring_xy(radius, th_g), z)
        if np.linalg.norm(obj - goal) < MIN_SEPARATION:
            rejected += 1
            continue
        objects.append(obj)
        goals.append(goal)
        thetas.append([th_o, th_g, z])
    return (np.asarray(objects), np.asarray(goals), np.asarray(thetas), rejected)


def generate_default_scenes(n, base_seed):
    """Experiment 1: untouched panda-gym sampling, indexed by env seed.

    Walks seeds upward from base_seed, keeping those whose scene is not degenerate.
    Returns the accepted seed list plus the positions each seed produces, so the
    identical scenes can be replayed by (and archived alongside) every arm.
    """
    env = gym.make(ENV_ID)
    seeds, objects, goals, rejected, seed = [], [], [], 0, base_seed
    while len(seeds) < n:
        obs, _ = env.reset(seed=seed)
        obj, goal = obs["achieved_goal"].copy(), obs["desired_goal"].copy()
        if np.linalg.norm(obj - goal) < MIN_SEPARATION:
            rejected += 1
        else:
            seeds.append(seed)
            objects.append(obj)
            goals.append(goal)
        seed += 1
    env.close()
    return np.asarray(seeds), np.asarray(objects), np.asarray(goals), rejected


def build_scene_table(n_default, n_ring):
    """Pre-generate every scene for every experiment. One rng, fixed seed."""
    rng = np.random.default_rng(SCENE_RNG_SEED)
    table, rejects = {}, {}

    seeds, obj, goal, rej = generate_default_scenes(n_default, BASE_SEED)
    table["exp1_seeds"] = seeds
    table["exp1_object"] = obj
    table["exp1_goal"] = goal
    rejects["exp1/default"] = rej

    for r in RADII:
        obj, goal, th, rej = generate_ring_scenes(rng, n_ring, r)
        key = f"exp2_r{r:.2f}"
        table[f"{key}_object"], table[f"{key}_goal"], table[f"{key}_theta"] = obj, goal, th
        rejects[f"exp2/r={r:.2f}"] = rej

    for z in GOAL_ZS:
        obj, goal, th, rej = generate_ring_scenes(rng, n_ring, HEIGHT_SWEEP_RADIUS, goal_z=z)
        key = f"exp3_z{z:.2f}"
        table[f"{key}_object"], table[f"{key}_goal"], table[f"{key}_theta"] = obj, goal, th
        rejects[f"exp3/z={z:.2f}"] = rej

    table["_rejected_keys"] = np.asarray(list(rejects.keys()))
    table["_rejected_counts"] = np.asarray(list(rejects.values()))
    return table, rejects


def save_scenes(table, path):
    np.savez_compressed(path, **table)


def load_scenes(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def load_actor(models_dir, ckpt, obs_dim, act_dim, device):
    """Load an actor plus ITS OWN normaliser stats.

    A missing normaliser is fatal: silently evaluating a normalised policy on raw
    observations looks exactly like a generalisation failure and would corrupt the
    whole comparison. BC writes "normaliser", SAC writes "Normalizer".
    """
    actor_path = Path(models_dir) / "Actor" / ckpt
    if not actor_path.is_file():
        raise FileNotFoundError(f"No actor checkpoint at {actor_path}")
    actor = ActorNetwork(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    candidates = [Path(models_dir) / d / ckpt for d in ("normaliser", "Normalizer")]
    norm_path = next((p for p in candidates if p.is_file()), None)
    if norm_path is None:
        raise FileNotFoundError(
            f"No normaliser stats for checkpoint '{ckpt}' in {[str(p.parent) for p in candidates]}"
        )
    normaliser = RunningMeanStd(obs_dim)
    normaliser.load_state_dict(torch.load(norm_path, map_location="cpu", weights_only=False))
    print(f"  actor      {actor_path}")
    print(f"  normaliser {norm_path}  (count={normaliser.count:.0f})")
    return actor, normaliser


class NetPolicy:
    """Deterministic tanh(mean) actor. No sampling anywhere."""

    def __init__(self, actor, normaliser, device):
        self.actor, self.normaliser, self.device = actor, normaliser, device

    def __call__(self, obs):
        fused = np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)
        fused = self.normaliser.normalise(fused)
        with torch.inference_mode():
            mean, _ = self.actor(torch.as_tensor(fused, device=self.device).unsqueeze(0))
            return torch.tanh(mean).squeeze(0).cpu().numpy().astype(np.float32)


class ExpertPolicy:
    """Wrapper so the per-episode reset reaches the expert's state machine.

    Passing the bare `expert.act` bound method skips reset(), leaks `phase` across
    episodes and silently crushes the control from ~100% to ~0-30%.
    """

    def __init__(self):
        self.expert = ScriptedExpert()

    def reset(self):
        self.expert.reset()

    def __call__(self, obs):
        return self.expert.act(obs)


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
class SceneInjector:
    """Replaces the task's samplers so reset() places object/goal from the table.

    Consumes zero env randomness, so every arm sees byte-identical scenes.
    """

    def __init__(self, env):
        self.task = env.unwrapped.task
        self.object_position = np.array([0.0, 0.0, CUBE_HALF])
        self.goal_position = np.array([0.0, 0.0, CUBE_HALF])
        self.task._sample_object = lambda: self.object_position.copy()
        self.task._sample_goal = lambda: self.goal_position.copy()

    def set(self, object_position, goal_position):
        self.object_position = np.asarray(object_position, dtype=float)
        self.goal_position = np.asarray(goal_position, dtype=float)


def rollout(env, policy, seed):
    """One episode. Returns (success, steps, final_distance, initial_obs)."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    initial = {"object": obs["achieved_goal"].copy(), "goal": obs["desired_goal"].copy()}
    success = bool(info.get("is_success", False))
    steps, done = 0, False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        steps += 1
        success = success or bool(info.get("is_success", False))
        done = bool(terminated or truncated)
    final_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    return success, steps, final_distance, initial


def run_bin(env, policy, objects, goals, injector, episodes, seed0):
    """Replay the first `episodes` scenes of one bin with one policy."""
    successes, distances = [], []
    for ep in range(episodes):
        injector.set(objects[ep], goals[ep])
        ok, _, dist, _ = rollout(env, policy, seed=seed0 + ep)
        successes.append(bool(ok))
        distances.append(dist)
    return np.asarray(successes), np.asarray(distances)


def run_default_bin(env, policy, seeds):
    """Experiment 1: replay the accepted default-sampling seeds."""
    successes, distances = [], []
    for seed in seeds:
        ok, _, dist, _ = rollout(env, policy, seed=int(seed))
        successes.append(bool(ok))
        distances.append(dist)
    return np.asarray(successes), np.asarray(distances)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson_interval(successes, n, z=1.959963984540054):
    """Wilson score 95% interval. Implemented directly -- only numpy/scipy assumed."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Clip: at p = 0 or 1 the exact bound is 0 or 1, but rounding lands a hair
    # outside, which later shows up as a negative error bar.
    return float(np.clip(centre - halfwidth, 0.0, 1.0)), float(np.clip(centre + halfwidth, 0.0, 1.0))


def mcnemar(a_success, b_success):
    """Exact McNemar on paired binary outcomes. b = a-only wins, c = b-only wins."""
    b = int(np.sum(a_success & ~b_success))
    c = int(np.sum(~a_success & b_success))
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return b, c, p


def summarise(successes, distances, rejected):
    n = len(successes)
    k = int(np.sum(successes))
    lo, hi = wilson_interval(k, n)
    fails = distances[~successes]
    return {
        "episodes": n,
        "successes": k,
        "success_rate": k / n if n else float("nan"),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "n_failures": int(len(fails)),
        "fail_dist_mean": float(np.mean(fails)) if len(fails) else float("nan"),
        "fail_dist_median": float(np.median(fails)) if len(fails) else float("nan"),
        "scenes_rejected": rejected,
    }


# --------------------------------------------------------------------------- #
# sanity checks
# --------------------------------------------------------------------------- #
def sanity(table, rejects, expert_episodes, settle_steps=5):
    """Position validity, replay determinism, and an expert feasibility probe."""
    print("\n" + "=" * 78)
    print("SANITY 1/3 -- sampled positions per bin")
    print("=" * 78)
    print(f"safe rect: x in [{SAFE_X[0]:+.2f}, {SAFE_X[1]:+.2f}]  "
          f"y in [{SAFE_Y[0]:+.2f}, {SAFE_Y[1]:+.2f}]\n")

    env = gym.make(ENV_ID)
    injector = SceneInjector(env)
    hold_still = lambda obs: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    all_ok = True

    bins = ([("exp2", f"r={r:.2f}", f"exp2_r{r:.2f}") for r in RADII]
            + [("exp3", f"z={z:.2f}", f"exp3_z{z:.2f}") for z in GOAL_ZS])

    print(f"{'bin':<14}{'obj x':>16}{'obj y':>16}{'goal z':>14}{'sep':>8}{'z after settle':>17}  ")
    print("-" * 78)
    for _, label, key in bins:
        obj, goal = table[f"{key}_object"], table[f"{key}_goal"]
        rect_ok = all(in_safe_rect(p[:2]) for p in obj) and all(in_safe_rect(p[:2]) for p in goal)
        sep = np.linalg.norm(obj - goal, axis=1)

        # Settle a handful of scenes: hold the arm still and confirm the cube stays put.
        settled = []
        for ep in range(min(5, len(obj))):
            injector.set(obj[ep], goal[ep])
            o, _ = env.reset(seed=BASE_SEED + ep)
            for _ in range(settle_steps):
                o, *_ = env.step(hold_still(o))
            settled.append(float(o["achieved_goal"][2]))
        settled = np.asarray(settled)
        z_ok = bool(np.all(np.abs(settled - CUBE_HALF) < 0.01))
        ok = rect_ok and z_ok and bool(np.all(sep >= MIN_SEPARATION))
        all_ok &= ok
        print(f"{label:<14}{obj[:, 0].min():>8.3f}..{obj[:, 0].max():<7.3f}"
              f"{obj[:, 1].min():>8.3f}..{obj[:, 1].max():<7.3f}"
              f"{goal[:, 2].min():>6.3f}..{goal[:, 2].max():<7.3f}"
              f"{sep.min():>8.3f}{settled.min():>9.3f}..{settled.max():<7.3f}"
              f"{'OK' if ok else 'FAIL':>4}")

    obj1, goal1 = table["exp1_object"], table["exp1_goal"]
    sep1 = np.linalg.norm(obj1 - goal1, axis=1)
    print(f"\nexp1 (default sampling): {len(table['exp1_seeds'])} scenes, "
          f"seeds {table['exp1_seeds'].min()}-{table['exp1_seeds'].max()}, "
          f"min separation {sep1.min():.3f} m, "
          f"obj r {np.linalg.norm(obj1[:, :2], axis=1).max():.3f} max")
    print(f"positions valid: {'OK' if all_ok else 'FAIL'}")
    print("\nscenes rejected at generation (initial separation < 0.05 m):")
    print("  " + "  ".join(f"{k}={v}" for k, v in rejects.items()))

    print("\n" + "=" * 78)
    print("SANITY 2/3 -- scene table replays identically across two loads")
    print("=" * 78)
    a, b = load_scenes("ood_scenes.npz"), load_scenes("ood_scenes.npz")
    identical = (a.keys() == b.keys()) and all(np.array_equal(a[k], b[k]) for k in a)
    matches_memory = all(np.array_equal(a[k], table[k]) for k in table if k in a)
    print(f"  load-vs-load identical: {'OK' if identical else 'FAIL'}  ({len(a)} arrays)")
    print(f"  load-vs-generated identical: {'OK' if matches_memory else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"SANITY 3/3 -- scripted expert feasibility probe ({expert_episodes} eps/bin)")
    print("=" * 78)
    expert = ExpertPolicy()
    probe, low = {}, []
    for _, label, key in bins:
        succ, _ = run_bin(env, expert, table[f"{key}_object"], table[f"{key}_goal"],
                          injector, expert_episodes, BASE_SEED)
        rate = float(np.mean(succ))
        probe[label] = rate
        if rate < 0.8:
            low.append((label, rate))
        print(f"  {label:<12} expert {rate:6.1%}")
    env.close()

    if low:
        print("\n  WARNING: expert below 80% in: "
              + ", ".join(f"{lbl} ({r:.0%})" for lbl, r in low))
    else:
        print("\n  expert near-ceiling in every bin -- all bins physically feasible")
    return all_ok and identical and matches_memory, probe, low


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def print_bin_summary(title, rows):
    print(f"\n  {title}")
    print(f"  {'bin':<12}{'arm':<10}{'n':>5}{'success':>10}{'95% CI':>18}"
          f"{'fail d̄':>10}{'fail med':>10}")
    print("  " + "-" * 73)
    for r in rows:
        ci = f"[{r['wilson_lo']:.2f}, {r['wilson_hi']:.2f}]"
        print(f"  {r['bin_label']:<12}{r['arm']:<10}{r['episodes']:>5}"
              f"{r['success_rate']:>10.1%}{ci:>18}"
              f"{r['fail_dist_mean']:>10.3f}{r['fail_dist_median']:>10.3f}")


def experiment_1(table, rejects, policies, episodes):
    """Default distribution -- the headline in-distribution number."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 -- default panda-gym distribution")
    print("=" * 78)
    started = time.perf_counter()
    env = gym.make(ENV_ID)
    seeds = table["exp1_seeds"][:episodes]
    rows, raw = [], {}
    for arm in ARMS:
        succ, dist = run_default_bin(env, policies[arm], seeds)
        raw[arm] = succ
        row = {"experiment": "exp1", "bin_label": "default", "bin_value": float("nan"),
               "arm": arm, **summarise(succ, dist, rejects["exp1/default"])}
        rows.append(row)
    env.close()
    b, c, p = mcnemar(raw["BC"], raw["SAC+HER"])
    for row in rows:
        row.update(mcnemar_b=b if row["arm"] == "BC" else "",
                   mcnemar_c=c if row["arm"] == "BC" else "",
                   mcnemar_p=p if row["arm"] == "BC" else "")
    elapsed = time.perf_counter() - started
    print_bin_summary(f"default distribution, {len(seeds)} paired episodes/arm", rows)
    print(f"\n  McNemar BC vs SAC+HER: b={b} (BC only) c={c} (SAC only) p={p:.4g}")
    print(f"  elapsed {elapsed:.1f}s")
    return rows, elapsed


def experiment_2(table, rejects, policies, episodes, expert_episodes):
    """Radius sweep -- the primary OOD axis."""
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 -- radius sweep (object and goal on a ring)")
    print("=" * 78)
    started = time.perf_counter()
    env = gym.make(ENV_ID)
    injector = SceneInjector(env)
    rows = []
    for r in RADII:
        key, label = f"exp2_r{r:.2f}", f"r={r:.2f}"
        obj, goal = table[f"{key}_object"], table[f"{key}_goal"]
        raw, bin_rows = {}, []
        for arm in ARMS:
            n = expert_episodes if arm == "expert" else episodes
            succ, dist = run_bin(env, policies[arm], obj, goal, injector, n, BASE_SEED)
            raw[arm] = succ
            bin_rows.append({"experiment": "exp2", "bin_label": label, "bin_value": r,
                             "arm": arm, **summarise(succ, dist, rejects[f"exp2/r={r:.2f}"])})
        b, c, p = mcnemar(raw["BC"], raw["SAC+HER"])
        for row in bin_rows:
            row.update(mcnemar_b=b if row["arm"] == "BC" else "",
                       mcnemar_c=c if row["arm"] == "BC" else "",
                       mcnemar_p=p if row["arm"] == "BC" else "")
        rows.extend(bin_rows)
        print(f"  r={r:.2f}  BC {bin_rows[0]['success_rate']:6.1%} | "
              f"SAC {bin_rows[1]['success_rate']:6.1%} | "
              f"expert {bin_rows[2]['success_rate']:6.1%} | McNemar p={p:.4g}")
    env.close()
    elapsed = time.perf_counter() - started
    print_bin_summary("radius sweep", rows)
    print(f"\n  elapsed {elapsed:.1f}s")
    return rows, elapsed


def experiment_3(table, rejects, policies, episodes, expert_episodes):
    """Goal-height sweep -- a second OOD axis, xy held in-distribution."""
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 3 -- goal height sweep (xy fixed at r={HEIGHT_SWEEP_RADIUS:.2f})")
    print("=" * 78)
    started = time.perf_counter()
    env = gym.make(ENV_ID)
    injector = SceneInjector(env)
    rows = []
    for z in GOAL_ZS:
        key, label = f"exp3_z{z:.2f}", f"z={z:.2f}"
        obj, goal = table[f"{key}_object"], table[f"{key}_goal"]
        raw, bin_rows = {}, []
        for arm in ARMS:
            n = expert_episodes if arm == "expert" else episodes
            succ, dist = run_bin(env, policies[arm], obj, goal, injector, n, BASE_SEED)
            raw[arm] = succ
            bin_rows.append({"experiment": "exp3", "bin_label": label, "bin_value": z,
                             "arm": arm, **summarise(succ, dist, rejects[f"exp3/z={z:.2f}"])})
        b, c, p = mcnemar(raw["BC"], raw["SAC+HER"])
        for row in bin_rows:
            row.update(mcnemar_b=b if row["arm"] == "BC" else "",
                       mcnemar_c=c if row["arm"] == "BC" else "",
                       mcnemar_p=p if row["arm"] == "BC" else "")
        rows.extend(bin_rows)
        print(f"  z={z:.2f}  BC {bin_rows[0]['success_rate']:6.1%} | "
              f"SAC {bin_rows[1]['success_rate']:6.1%} | "
              f"expert {bin_rows[2]['success_rate']:6.1%} | McNemar p={p:.4g}")
    env.close()
    elapsed = time.perf_counter() - started
    print_bin_summary("goal height sweep", rows)
    print(f"\n  elapsed {elapsed:.1f}s")
    return rows, elapsed


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["experiment", "bin_label", "bin_value", "arm", "episodes", "successes",
              "success_rate", "wilson_lo", "wilson_hi", "n_failures", "fail_dist_mean",
              "fail_dist_median", "scenes_rejected", "mcnemar_b", "mcnemar_c", "mcnemar_p"]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[out] wrote {path} ({len(rows)} rows)")


def _sweep_plot(rows, experiment, xlabel, title, boundaries, path, legend_loc="lower left"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Stagger the caption rows so two nearby boundaries cannot overlap.
    for row, (x, label, colour) in enumerate(boundaries):
        ax.axvline(x, color=colour, ls=(0, (4, 3)), lw=1.4, zorder=1)
        ax.annotate(label, xy=(x, 1.09 + 0.08 * row), ha="center", va="bottom",
                    fontsize=7.5, color=colour, annotation_clip=False)

    for arm in ARMS:
        sel = [r for r in rows if r["experiment"] == experiment and r["arm"] == arm]
        xs = np.array([r["bin_value"] for r in sel])
        ys = np.array([r["success_rate"] for r in sel])
        lo = np.maximum(ys - np.array([r["wilson_lo"] for r in sel]), 0.0)
        hi = np.maximum(np.array([r["wilson_hi"] for r in sel]) - ys, 0.0)
        is_expert = arm == "expert"
        name = "scripted expert (feasibility ceiling)" if is_expert else arm
        ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOUR[arm], lw=2.0,
                    ls="--" if is_expert else "-", marker=ARM_MARKER[arm], ms=7,
                    mec=SURFACE, mew=1.5, capsize=3.5, elinewidth=1.4,
                    label=name, zorder=4 if not is_expert else 3)

    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel("success rate", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11.5, pad=52, loc="left")
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    # Opaque patch in the surface colour so a gridline or boundary rule crossing
    # behind the legend cannot cut through the label text.
    ax.legend(fontsize=8.5, loc=legend_loc, labelcolor=INK, frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=1.0).set_zorder(6)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"[out] wrote {path}")


def make_plots(rows):
    _sweep_plot(
        rows, "exp2",
        "distance of object and goal from table centre, r (m)",
        "OOD generalisation vs radial distance — PandaPickAndPlace-v3",
        [(0.15, "edge of training square (r=0.15)", "#008300"),
         (0.2121, "fully outside support (r=0.212)", "#d03b3b")],
        "ood_radius.png",
    )
    _sweep_plot(
        rows, "exp3",
        "goal height above table, z (m)   [xy fixed at r=0.10]",
        "OOD generalisation vs goal height — PandaPickAndPlace-v3",
        [(TRAIN_Z_CEILING, f"training ceiling (z={TRAIN_Z_CEILING:.2f})", "#d03b3b")],
        "ood_height.png",
        legend_loc="center",   # SAC sits flat on 0% and BC on ~100%: the middle is empty
    )


def describe_setup(device):
    import scipy
    lines = [
        f"python           {sys.version.split()[0]} ({sys.executable})",
        f"platform         {platform.platform()}",
        f"processor        {platform.processor()}",
        f"torch            {torch.__version__}",
        f"torch.version.cuda {torch.version.cuda}",
        f"cuda available   {torch.cuda.is_available()}",
        f"device used      {device}",
        f"numpy            {np.__version__}",
        f"scipy            {scipy.__version__}",
        f"gymnasium        {gym.__version__}",
        f"panda_gym        {panda_gym.__version__}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu              {torch.cuda.get_device_name(0)}")
    elif torch.version.cuda is None:
        lines.append("NOTE: torch is a CPU-only build (torch.version.cuda is None) -- "
                     "the GPU cannot be used regardless of hardware.")
    return lines


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bc-dir", default=f"models/BC_{ENV_ID}")
    parser.add_argument("--bc-checkpoint", default="95_best")
    parser.add_argument("--sac-dir", default=f"models/SAC_{ENV_ID}")
    parser.add_argument("--sac-checkpoint", default="291904_interrupted")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes/bin for BC and SAC")
    parser.add_argument("--expert-episodes", type=int, default=50, help="Episodes/bin for the expert control")
    parser.add_argument("--exp1-episodes", type=int, default=200, help="Episodes for experiment 1 (all arms)")
    parser.add_argument("--probe-episodes", type=int, default=10, help="Episodes/bin for the sanity expert probe")
    parser.add_argument("--scenes", default="ood_scenes.npz")
    parser.add_argument("--csv", default="ood_results.csv")
    parser.add_argument("--replot", action="store_true", help="Redraw the plots from an existing --csv and exit")
    parser.add_argument("--sanity", action="store_true", help="Run sanity checks only and exit")
    parser.add_argument("--skip-sanity", action="store_true", help="Go straight to the experiments")
    args = parser.parse_args()

    total_started = time.perf_counter()

    if args.replot:
        with open(args.csv, newline="", encoding="utf-8") as fh:
            rows = [{**r, "bin_value": float(r["bin_value"]),
                     "success_rate": float(r["success_rate"]),
                     "wilson_lo": float(r["wilson_lo"]), "wilson_hi": float(r["wilson_hi"])}
                    for r in csv.DictReader(fh)]
        make_plots(rows)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    for line in describe_setup(device):
        print("  " + line)

    print("\n" + "=" * 78)
    print("SCENE TABLE")
    print("=" * 78)
    table, rejects = build_scene_table(args.exp1_episodes, args.episodes)
    save_scenes(table, args.scenes)
    total_scenes = args.exp1_episodes + args.episodes * (len(RADII) + len(GOAL_ZS))
    print(f"  rng            np.random.default_rng({SCENE_RNG_SEED})")
    print(f"  arc            [{np.rad2deg(THETA_LO):.1f}, {np.rad2deg(THETA_HI):.1f}] deg "
          f"({np.rad2deg(THETA_HI - THETA_LO):.1f} deg wide)")
    print(f"  scenes         {total_scenes} across {1 + len(RADII) + len(GOAL_ZS)} bins")
    print(f"  rejected       {sum(rejects.values())} (initial separation < {MIN_SEPARATION} m)")
    print(f"  archived to    {args.scenes}")

    if not args.skip_sanity:
        ok, _, low = sanity(table, rejects, args.probe_episodes)
        if low:
            print("\nHALTING: the scripted expert is below 80% in the bins listed above.")
            print("A low expert score means those scenes are physically infeasible and the")
            print("policy comparison there would be meaningless. Investigate before sweeping.")
            return 1
        if not ok:
            print("\nHALTING: a position/replay sanity check failed.")
            return 1
    if args.sanity:
        return 0

    print("\n" + "=" * 78)
    print("POLICIES")
    print("=" * 78)
    probe_env = gym.make(ENV_ID)
    obs_dim = (probe_env.observation_space["observation"].shape[0]
               + probe_env.observation_space["desired_goal"].shape[0])
    act_dim = probe_env.action_space.shape[0]
    probe_env.close()
    print("BC")
    bc_actor, bc_norm = load_actor(args.bc_dir, args.bc_checkpoint, obs_dim, act_dim, device)
    print("SAC+HER")
    sac_actor, sac_norm = load_actor(args.sac_dir, args.sac_checkpoint, obs_dim, act_dim, device)
    policies = {
        "BC": NetPolicy(bc_actor, bc_norm, device),
        "SAC+HER": NetPolicy(sac_actor, sac_norm, device),
        "expert": ExpertPolicy(),
    }

    rows = []
    r1, t1 = experiment_1(table, rejects, policies, args.exp1_episodes)
    rows += r1
    r2, t2 = experiment_2(table, rejects, policies, args.episodes, args.expert_episodes)
    rows += r2
    r3, t3 = experiment_3(table, rejects, policies, args.episodes, args.expert_episodes)
    rows += r3

    write_csv(rows, args.csv)
    make_plots(rows)

    total = time.perf_counter() - total_started
    print(f"\n[time] exp1 {t1:.1f}s | exp2 {t2:.1f}s | exp3 {t3:.1f}s | total {total:.1f}s "
          f"({total / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
