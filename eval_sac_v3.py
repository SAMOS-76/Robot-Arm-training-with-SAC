"""
Extends the BC/SAC+HER/expert OOD comparison in eval_ood.py with a fourth arm:
SAC+HER trained with the replay buffer seeded from `expert_demos.npz`
(SAC_agent_HER_panda.py: load_demonstrations).

WHY A SEPARATE SCRIPT: eval_ood.py's ARMS list and its McNemar wiring are hardcoded
to the BC-vs-SAC+HER pair that the original report compares, and that report is a
committed deliverable. Rather than rewrite that pipeline for a 4th arm, this script
reuses its lower-level building blocks (load_actor, NetPolicy, run_bin, run_default_bin,
wilson_interval, mcnemar, the plot styling) against the SAME archived scene table
(ood_scenes.npz), so results are directly comparable to ood_results.csv without
re-running BC / old-SAC / expert.

WHAT THIS ANSWERS: demo-seeding the replay buffer (Nair et al. 2018 style) made
SAC+HER jump from ~0% lift rate to ~78% training success in ~60k env-steps -- fast
enough to be suspicious, since BC was trained on the exact same 200 demo episodes
this checkpoint's buffer was seeded with. Two questions follow:
  1. Does it generalise OOD like BC (i.e. did SAC+HER converge to just replaying the
     demonstrations), or does it show a different profile (i.e. the online RL phase
     changed the policy into something else)?
  2. Does demo-seeding actually beat plain SAC+HER (no demos), or does the fast
     climb just reflect an easier optimisation problem that plateaus lower?
Pairwise McNemar on all three learned arms, all paired on the same scenes, answers
both directly instead of by eyeballing curves.

    python eval_sac_v3.py --sac-demo-dir models/SAC_PickPlace_v3 --sac-demo-checkpoint 4999_solved
"""
import argparse
import csv
import itertools
import sys
import time

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch

import eval_ood as base
from eval_ood import (ENV_ID, RADII, GOAL_ZS, HEIGHT_SWEEP_RADIUS, BASE_SEED,
                      SceneInjector, NetPolicy, ExpertPolicy, load_actor, load_scenes,
                      run_bin, run_default_bin, wilson_interval, mcnemar, summarise,
                      describe_setup)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Slot 4 (yellow) sits adjacent to slot 2 (orange) in the shared palette and is a
# documented bad pair there; violet (slot 7) is used instead for the 4th series so
# the two SAC arms -- the pair most likely to be compared directly -- stay well apart.
ARM_COLOUR_V3 = {"BC": "#2a78d6", "SAC+HER (no demo)": "#eb6834",
                 "SAC+HER (demo-seeded)": "#4a3aa7", "expert": "#1baf7a"}
ARM_MARKER_V3 = {"BC": "o", "SAC+HER (no demo)": "s", "SAC+HER (demo-seeded)": "D", "expert": "^"}
LEARNED_ARMS = ["BC", "SAC+HER (no demo)", "SAC+HER (demo-seeded)"]
ARMS_V3 = LEARNED_ARMS + ["expert"]


def run_experiment(env, injector, table, rejects_lookup, policies, arm_names, exp_key, bins,
                   episodes, expert_episodes, is_exp1=False):
    """One experiment across all arms; returns summary rows + a raw success dict per bin.

    `env` must be an UNPATCHED env (default panda-gym sampling) when is_exp1=True, and
    the SceneInjector-wrapped env otherwise -- see the two-env setup in main().
    """
    rows, raw_by_bin = [], {}
    for bin_key, label, bin_value, reject_key in bins:
        raw = {}
        rejected = int(rejects_lookup.get(reject_key, 0))
        for arm in arm_names:
            if is_exp1:
                succ, dist = run_default_bin(env, policies[arm], table["exp1_seeds"][:episodes])
            else:
                n = expert_episodes if arm == "expert" else episodes
                obj, goal = table[f"{bin_key}_object"], table[f"{bin_key}_goal"]
                succ, dist = run_bin(env, policies[arm], obj, goal, injector, n, BASE_SEED)
            raw[arm] = succ
            rows.append({"experiment": exp_key, "bin_label": label, "bin_value": bin_value,
                        "arm": arm, **summarise(succ, dist, rejected)})
        raw_by_bin[label] = raw
    return rows, raw_by_bin


def pairwise_mcnemar(raw_by_bin, exp_key):
    """All C(3,2)=3 pairwise McNemar tests among the learned arms, per bin."""
    out = []
    for label, raw in raw_by_bin.items():
        bin_value = None
        for a, b in itertools.combinations(LEARNED_ARMS, 2):
            b_cnt, c_cnt, p = mcnemar(raw[a], raw[b])
            out.append({"experiment": exp_key, "bin_label": label, "arm_a": a, "arm_b": b,
                       "b_a_only": b_cnt, "c_b_only": c_cnt, "p_value": p})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bc-dir", default=f"models/BC_{ENV_ID}")
    parser.add_argument("--bc-checkpoint", default="95_best")
    parser.add_argument("--sac-nodemo-dir", default=f"models/SAC_{ENV_ID}")
    parser.add_argument("--sac-nodemo-checkpoint", default="291904_interrupted")
    parser.add_argument("--sac-demo-dir", required=True)
    parser.add_argument("--sac-demo-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--expert-episodes", type=int, default=50)
    parser.add_argument("--exp1-episodes", type=int, default=200)
    parser.add_argument("--scenes", default="ood_scenes.npz")
    parser.add_argument("--csv", default="ood_results_v3.csv")
    parser.add_argument("--mcnemar-csv", default="ood_mcnemar_v3.csv")
    parser.add_argument("--plot-suffix", default="v3",
                        help="Plots are written to ood_radius_<suffix>.png / ood_height_<suffix>.png "
                             "-- set a fresh suffix per checkpoint so repeat runs don't overwrite each other's plots.")
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    for line in describe_setup(device):
        print("  " + line)

    table = load_scenes(args.scenes)
    rejects_lookup = dict(zip(table["_rejected_keys"].tolist(), table["_rejected_counts"].tolist()))
    print(f"\n[scenes] loaded {args.scenes} ({len(table)} arrays)")

    # Two separate envs, mirroring eval_ood.py's own structure: experiment_1 needs
    # UNPATCHED default panda-gym sampling, while experiments 2/3 need the injector.
    # SceneInjector permanently overwrites the task's _sample_object/_sample_goal to
    # a constant (0,0,CUBE_HALF) for both object and goal until .set() is called per
    # episode -- sharing one env between exp1 and exp2/3 would make "default sampling"
    # silently return that degenerate zero-separation scene on every reset.
    default_env = gym.make(ENV_ID)
    obs_dim = (default_env.observation_space["observation"].shape[0]
              + default_env.observation_space["desired_goal"].shape[0])
    act_dim = default_env.action_space.shape[0]

    env = gym.make(ENV_ID)
    injector = SceneInjector(env)

    print("\n" + "=" * 78)
    print("POLICIES")
    print("=" * 78)
    print("BC")
    bc_a, bc_n = load_actor(args.bc_dir, args.bc_checkpoint, obs_dim, act_dim, device)
    print("SAC+HER (no demo)")
    nodemo_a, nodemo_n = load_actor(args.sac_nodemo_dir, args.sac_nodemo_checkpoint, obs_dim, act_dim, device)
    print("SAC+HER (demo-seeded)")
    demo_a, demo_n = load_actor(args.sac_demo_dir, args.sac_demo_checkpoint, obs_dim, act_dim, device)
    policies = {
        "BC": NetPolicy(bc_a, bc_n, device),
        "SAC+HER (no demo)": NetPolicy(nodemo_a, nodemo_n, device),
        "SAC+HER (demo-seeded)": NetPolicy(demo_a, demo_n, device),
        "expert": ExpertPolicy(),
    }

    all_rows, all_mcnemar = [], []

    print("\n" + "=" * 78)
    print("EXPERIMENT 1 -- default distribution")
    print("=" * 78)
    rows, raw = run_experiment(default_env, injector, table, rejects_lookup, policies, ARMS_V3, "exp1",
                               [("exp1", "default", float("nan"), "exp1/default")],
                               args.exp1_episodes, args.exp1_episodes, is_exp1=True)
    all_rows += rows
    mc = pairwise_mcnemar(raw, "exp1")
    all_mcnemar += mc
    for r in rows:
        print(f"  {r['arm']:<26} {r['success_rate']:6.1%}  [{r['wilson_lo']:.2f}, {r['wilson_hi']:.2f}]")
    for m in mc:
        print(f"  McNemar {m['arm_a']} vs {m['arm_b']}: b={m['b_a_only']} c={m['c_b_only']} p={m['p_value']:.4g}")

    print("\n" + "=" * 78)
    print("EXPERIMENT 2 -- radius sweep")
    print("=" * 78)
    bins = [(f"exp2_r{r:.2f}", f"r={r:.2f}", r, f"exp2/r={r:.2f}") for r in RADII]
    rows, raw = run_experiment(env, injector, table, rejects_lookup, policies, ARMS_V3, "exp2", bins,
                               args.episodes, args.expert_episodes)
    all_rows += rows
    mc = pairwise_mcnemar(raw, "exp2")
    all_mcnemar += mc
    for r_val in RADII:
        label = f"r={r_val:.2f}"
        vals = {r["arm"]: r["success_rate"] for r in rows if r["bin_label"] == label}
        print(f"  {label}  " + " | ".join(f"{a}={vals[a]:.1%}" for a in ARMS_V3))
    for m in mc:
        print(f"    [{m['bin_label']}] {m['arm_a']} vs {m['arm_b']}: "
              f"b={m['b_a_only']} c={m['c_b_only']} p={m['p_value']:.4g}")

    print("\n" + "=" * 78)
    print("EXPERIMENT 3 -- goal height sweep")
    print("=" * 78)
    bins = [(f"exp3_z{z:.2f}", f"z={z:.2f}", z, f"exp3/z={z:.2f}") for z in GOAL_ZS]
    rows, raw = run_experiment(env, injector, table, rejects_lookup, policies, ARMS_V3, "exp3", bins,
                               args.episodes, args.expert_episodes)
    all_rows += rows
    mc = pairwise_mcnemar(raw, "exp3")
    all_mcnemar += mc
    for z_val in GOAL_ZS:
        label = f"z={z_val:.2f}"
        vals = {r["arm"]: r["success_rate"] for r in rows if r["bin_label"] == label}
        print(f"  {label}  " + " | ".join(f"{a}={vals[a]:.1%}" for a in ARMS_V3))
    for m in mc:
        print(f"    [{m['bin_label']}] {m['arm_a']} vs {m['arm_b']}: "
              f"b={m['b_a_only']} c={m['c_b_only']} p={m['p_value']:.4g}")

    env.close()
    default_env.close()

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["experiment", "bin_label", "bin_value", "arm", "episodes", "successes",
                     "success_rate", "wilson_lo", "wilson_hi", "n_failures", "fail_dist_mean",
                     "fail_dist_median", "scenes_rejected"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[out] wrote {args.csv} ({len(all_rows)} rows)")

    with open(args.mcnemar_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "bin_label", "arm_a", "arm_b",
                                          "b_a_only", "c_b_only", "p_value"])
        w.writeheader()
        w.writerows(all_mcnemar)
    print(f"[out] wrote {args.mcnemar_csv} ({len(all_mcnemar)} rows)")

    # Reuse eval_ood's plot styling exactly (same fonts/spacers/legend rules) by
    # temporarily pointing its arm registry at this script's 4 arms.
    orig_arms, orig_colour, orig_marker = base.ARMS, base.ARM_COLOUR, base.ARM_MARKER
    base.ARMS, base.ARM_COLOUR, base.ARM_MARKER = ARMS_V3, ARM_COLOUR_V3, ARM_MARKER_V3
    try:
        base._sweep_plot(
            all_rows, "exp2",
            "distance of object and goal from table centre, r (m)",
            "OOD generalisation vs radial distance — 4-arm comparison",
            [(0.15, "edge of training square (r=0.15)", "#008300"),
             (0.2121, "fully outside support (r=0.212)", "#d03b3b")],
            f"ood_radius_{args.plot_suffix}.png",
        )
        base._sweep_plot(
            all_rows, "exp3",
            "goal height above table, z (m)   [xy fixed at r=0.10]",
            "OOD generalisation vs goal height — 4-arm comparison",
            [(base.TRAIN_Z_CEILING, f"training ceiling (z={base.TRAIN_Z_CEILING:.2f})", "#d03b3b")],
            f"ood_height_{args.plot_suffix}.png",
            legend_loc="center",
        )
    finally:
        base.ARMS, base.ARM_COLOUR, base.ARM_MARKER = orig_arms, orig_colour, orig_marker

    print(f"\n[time] total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
