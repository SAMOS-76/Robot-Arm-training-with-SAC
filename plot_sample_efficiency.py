"""
Sample-efficiency plot: in-distribution online success rate vs training cost, parsed
directly from training logs (not the OOD eval -- this is in-distribution, during-training
performance, a different axis from ood_radius_final.png).

Three arms, two cost axes:

  env-steps   -- environment interaction consumed. The SAC arms burn this online;
                 BC does not interact with the env at all, so its entire data cost is
                 the fixed 3533-transition demo set (the same one seeded into the
                 demo-seeded SAC run). BC therefore has no curve on this axis -- it is
                 a single point, drawn as a marker + reference line, NOT interpolated.
  wall-clock  -- seconds of training. Shared and meaningful for all three arms.

The two SAC curves are trailing-100-episode online success (dense, every 10 episodes).
BC's is a 20-episode deterministic rollout eval every --save-epochs epochs, so it is
sparse and noisy (n=20 -> a 100% reading has a Wilson CI of [83.9%, 100%]); points are
plotted where they were measured with no interpolation between them.

    python plot_sample_efficiency.py [--out sample_efficiency.png]
"""
import argparse
import re

import numpy as np

# (log path, colour, linestyle, format) -- colours match ood_radius_final.png
ARM_LOGS = {
    "BC": ("logs/bc_matched.log", "#2a78d6", "-", "bc"),
    "SAC+HER (no demo)": ("logs/sac_nodemo_matched.log", "#eb6834", "-", "sac"),
    "SAC+HER (demo-seeded)": ("logs/sac_v4.log", "#4a3aa7", "-", "sac"),
}

# BC's total data cost: the demo set it is fit on. Matches the "3533 transitions"
# seeded into the demo-seeded SAC run, so the two are directly comparable.
BC_DEMO_TRANSITIONS = 3533

SAC_RE = re.compile(
    r"\[(\d+):(\d+):(\d+)\] Step: (\d+) \| SPS: \d+ \| Ep: (\d+)\s*\n.*?Succ\(100\)=([\d.]+|nan)"
)
# Externally-added "HH:MM:SS " prefix (train_bc.py prints no timestamps itself).
BC_EPOCH_RE = re.compile(r"^(\d+):(\d+):(\d+) Epoch: (\d+)$")
BC_SUCC_RE = re.compile(r"^(\d+):(\d+):(\d+) Success rate: ([\d.]+)%$")
BC_START_RE = re.compile(r"^(\d+):(\d+):(\d+) train transitions:")

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def parse_sac_log(path):
    """-> [(env_steps, wall_seconds, episodes, success_rate)]. Step: is ALREADY total
    env-steps (verified: sac_v4.log ends at Step: 3599916 == 300000 global x 12 envs),
    so it must NOT be multiplied by n_envs again."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    for h, m, s, step, ep, succ in SAC_RE.findall(txt):
        if succ == "nan":
            continue
        wall = int(h) * 3600 + int(m) * 60 + int(s)
        rows.append((int(step), wall, int(ep), float(succ)))
    return rows


def parse_bc_log(path):
    """-> [(env_steps, wall_seconds, epoch, success_rate)]. env_steps is the constant
    demo-set size: BC consumes zero online interaction, so its data cost does not grow
    with training. Wall-clock is relative to the start of train()."""
    rows, epoch, t0 = [], None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if t0 is None:
            m = BC_START_RE.match(line)
            if m:
                t0 = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            continue
        m = BC_EPOCH_RE.match(line)
        if m:
            epoch = int(m[4])
            continue
        m = BC_SUCC_RE.match(line)
        if m:
            wall = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) - t0
            rows.append((BC_DEMO_TRANSITIONS, wall, epoch, float(m[4]) / 100.0))
    return rows


def threshold_table(rows, thresholds=(0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)):
    """First crossing of each threshold, in log order."""
    out, hit = [], set()
    for env_steps, wall, counter, succ in rows:
        for t in thresholds:
            if t not in hit and succ >= t:
                hit.add(t)
                out.append((t, env_steps, wall, counter))
    return out


def fmt_wall(sec):
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def style(ax, xlabel, title):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.grid(axis="x", which="major", color=GRID, lw=0.6, zorder=0, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_efficiency.png", help="Output plot filename")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    hdr = f"{'arm':<24}{'threshold':>10}{'env-steps':>12}{'wall-clock':>12}{'ep/epoch':>10}"
    print(hdr)
    print("-" * len(hdr))

    for arm, (path, colour, ls, kind) in ARM_LOGS.items():
        rows = (parse_bc_log if kind == "bc" else parse_sac_log)(path)
        if not rows:
            print(f"{arm:<24}  (no parsable rows in {path})")
            continue
        x_steps = np.array([r[0] for r in rows])
        wall = np.array([r[1] for r in rows])
        succ = np.array([r[3] for r in rows])

        if kind == "bc":
            # Constant x on the env-steps axis: one point, plus a reference line at the
            # level BC actually reaches. Drawing a curve here would imply BC spends
            # increasing env-steps, which it does not.
            # Anchor at the final reading, not max(): these are n=20 evals, so the max
            # over 20 of them is biased upward (the 200-episode OOD eval puts BC's true
            # in-distribution rate at 95%, Wilson [91.0, 97.3]).
            ax1.axhline(succ[-1], color=colour, ls=":", lw=1.4, alpha=0.75, zorder=2)
            ax1.plot(x_steps[0], succ[-1], marker="o", ms=7, color=colour,
                     label=f"{arm} (fixed {BC_DEMO_TRANSITIONS:,}-transition demo set, n=20 eval)", zorder=3)
            ax2.plot(wall, succ, color=colour, ls=ls, lw=1.6, marker="o", ms=3.5, label=arm)
        else:
            ax1.plot(x_steps, succ, color=colour, ls=ls, lw=1.8, label=arm)
            ax2.plot(wall, succ, color=colour, ls=ls, lw=1.8, label=arm)

        for t, env_steps, w, counter in threshold_table(rows):
            steps_s = f"{env_steps:,}*" if kind == "bc" else f"{env_steps:,}"
            print(f"{arm:<24}{t:>10.0%}{steps_s:>12}{fmt_wall(w):>12}{counter:>10,}")
        print(f"{arm:<24}{'final':>10}{f'{x_steps[-1]:,}':>12}{fmt_wall(wall[-1]):>12}"
              f"{rows[-1][2]:>10,}  (succ={succ[-1]:.2f}, max={succ.max():.2f})")
        print("-" * len(hdr))

    print("* BC consumes no online env-steps; the figure is its fixed demo-set size, "
          "constant across all thresholds.")

    style(ax1, "env-steps consumed (log scale)", "Sample efficiency — success vs environment interaction")
    style(ax2, "wall-clock seconds (log scale)", "Wall-clock efficiency — success vs training time")
    ax1.set_ylabel("in-distribution online success rate", color=INK, fontsize=10)
    ax1.legend(fontsize=8.5, loc="center left", frameon=False, labelcolor=INK)
    ax2.legend(fontsize=8.5, loc="lower right", frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
