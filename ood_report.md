# Out-of-distribution generalisation: behaviour cloning vs SAC+HER on `PandaPickAndPlace-v3`

A paired comparison of two learned actors — a behaviour-cloning (BC) policy and a
SAC+HER policy — measuring how each degrades as object and goal positions move outside
the distribution they were trained on. A scripted expert with privileged state runs on
the identical scenes as a feasibility ceiling.

**Headline result.** On the training distribution BC succeeds on 95.0% of episodes and
SAC+HER on 52.0%. The gap is not a matter of degree: the two policies learned
*different strategies*. BC learned to pick up and carry the cube; the SAC+HER checkpoint
learned only to **push it along the table** — across the 120 episodes in which cube height
was instrumented it never once raised the cube above 0.023 m (its resting height is 0.020 m).
Every downstream difference in this report follows from that.

---

## 1. Methodology

### 1.1 Environment

`PandaPickAndPlace-v3` from panda-gym 3.0.7: a 7-DoF Franka Panda with a parallel
gripper, a 0.04 m cube, and a goal that may be on the table or in the air. Sparse reward;
success is the cube's centre within `distance_threshold = 0.05` m of the goal. Episodes
are capped at 50 steps by a `TimeLimit` wrapper.

`core.py:321` sets `terminated = bool(self.task.is_success(...))` and `truncated = False`,
so the environment **terminates the instant success occurs** and the 50-step cap is the
only other exit. Consequence: every failed episode runs the full 50 steps, so the final
‖achieved_goal − desired_goal‖ of a failure is a clean "near miss vs totally lost" signal.
This report uses it as a secondary metric throughout.

### 1.2 The training distribution, and what counts as outside it

`panda_gym/envs/tasks/pick_and_place.py` lines 24–27 and 70–84 define the support:

| quantity | distribution |
|---|---|
| object xy | `U(−0.15, +0.15)²`, z fixed at 0.02 |
| goal xy | `U(−0.15, +0.15)²` |
| goal z | `U(0, 0.2)` above the cube centre, forced flat to 0.02 with probability 0.3 |

So the training support in xy is a **±0.15 m square**: radius ≤ 0.15 is fully inside it,
and radius > 0.15·√2 = **0.212** is fully outside it. In height, the goal centre ceiling is
0.02 + 0.20 = **0.22 m** absolute. (The task brief quotes 0.20 as the ceiling; that is the
`goal_z_range` parameter, to which the cube's 0.02 half-extent is added. Both readings put
the same three height bins outside the support, so no conclusion turns on the difference.
Plots mark the exact 0.22.)

### 1.3 Overriding the sampler

These ranges are **not settable through `gym.make`** — `panda_tasks.py:103` hardcodes
`task = PickAndPlace(sim, reward_type=reward_type)` and never forwards `obj_xy_range`,
`goal_xy_range` or `goal_z_range`. However `_sample_object` and `_sample_goal` read their
range attributes off `self` at every reset, so **replacing the two bound methods on the
task instance** is sufficient. No subclassing, no gym-registry changes:

```python
env.unwrapped.task._sample_object = my_fn
env.unwrapped.task._sample_goal   = my_fn
```

The replacements installed here return positions from a pre-generated table rather than
drawing fresh randomness (§1.5).

### 1.4 Geometry: a fixed safe arc

Swept positions are generated as `centre + r·(cos θ, sin θ)` about the table centre, not
by panda-gym's uniform square.

The table top spans x ∈ [−0.85, 0.25], y ∈ [−0.35, 0.35] (`pybullet.py`,
`create_table(length=1.1, width=0.7, height=0.4, x_offset=−0.3)`). With the cube's 0.02
half-extent and a 0.03 edge margin, a cube centre is safe in
**x ∈ [−0.80, 0.20], y ∈ [−0.30, 0.30]**.

Because the table ends at x = +0.25, a full 360° ring is impossible beyond r ≈ 0.20.
Rejecting samples per-radius would make the angular distribution co-vary with radius and
**confound distance with direction**. Instead θ is restricted to the *same arc at every
radius*:

> **θ ∈ [48.2°, 311.8°]** — 263.6° wide, excluding a 96.4° wedge around +x.

derived from cos θ ≤ 0.20 / r_max with r_max = 0.30. r_max is capped at 0.30 because at
0.35 the y-bound also binds and the safe region splits into two disconnected arcs.

**Tradeoff, stated plainly:** this costs coverage of the +x region in *every* swept bin,
including the in-distribution ones, so Experiments 2 and 3 do not sample the training
support uniformly. Experiment 1 exists to recover that — it uses untouched panda-gym
sampling over the full square.

### 1.5 The scene table (what makes this paired)

**One** table of scenes is generated before any policy runs, from
`np.random.default_rng(12345)`, and every arm replays it identically. For each
(experiment, bin, episode) it stores θ_obj, θ_goal, goal z, and the resulting object and
goal xyz. Archived to **`ood_scenes.npz`** (44 arrays).

Any scene whose initial object–goal separation is < 0.05 m is **rejected and resampled**:
it is already inside the success threshold and would auto-succeed on step 1. 71 scenes
were rejected in total; per-bin counts are in `ood_results.csv`.

Letting each policy's rollout draw from the env RNG would break this. Two policies that
consume different numbers of RNG draws would silently desynchronise the scenes and the
pairing would be lost without any error surfacing. The injected samplers consume **zero**
environment randomness, so the pairing is exact by construction.

Experiment 1 is paired differently but equivalently: `core.py:280` does
`self.task.np_random, seed = seeding.np_random(seed)`, so the env seed fully determines
placement, and nothing during an episode consumes task RNG. All arms therefore replay the
same seed list. Seeds walk upward from **20000**, skipping degenerate scenes (6 skipped,
final seed 20205). A base seed ≥ 200 is mandatory here: the expert demonstrations in
`expert_demos.npz` were recorded on seeds 0–199 (`scripted_expert.py:222–223`), so any
lower base seed would replay the exact scenes BC was trained on.

### 1.6 Arms

| arm | checkpoint | normaliser | count |
|---|---|---|---|
| BC | `models/BC_PandaPickAndPlace-v3/Actor/95_best` | `normaliser/95_best` | 3 164 |
| SAC+HER | `models/SAC_PandaPickAndPlace-v3/Actor/291904_interrupted` | `Normalizer/291904_interrupted` | 2 335 240 |
| scripted expert | n/a — `ScriptedExpert` from `scripted_expert.py` | n/a | n/a |

Note the two different directory spellings (`normaliser` vs `Normalizer`); the loader
looks up both and treats a **missing normaliser as a hard error**. A silent fallback to
raw observations would look exactly like a generalisation failure and would corrupt the
comparison. The counts above are printed at load time and match the expected anchors.

All policies act **deterministically**: `action = torch.tanh(mean)`, never `rsample()` or
`Normal(...).sample()`. Each uses its own checkpoint's normaliser stats. SAC's
`ActorNetwork` is architecturally identical to BC's (3×256 MLP, mean + log-σ heads), so
`B_cloning.ActorNetwork` loads both.

The scripted expert is a **feasibility control, not a competitor**. It uses privileged
state, so wherever it succeeds the scene is physically reachable and any policy failure
there is genuine generalisation failure rather than the workspace running out. It is
reported as a ceiling line on every plot.

`ScriptedExpert` is a state machine that **must** have `.reset()` called between episodes.
Passing the bare `expert.act` bound method as a policy callable skips the reset, leaks
`phase` across episodes and silently crushes the control from ~100% to ~0–30%. It is
wrapped in an object exposing both `__call__` and `reset`.

### 1.7 Experiments

| # | design | episodes / bin |
|---|---|---|
| 1 | **Default distribution.** Untouched panda-gym sampling. | 200 per arm |
| 2 | **Radius sweep.** r ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}; object and goal both on the ring at *independent* safe-arc angles; goal z from panda-gym's default distribution so only xy radius varies. | 100 BC/SAC, 50 expert |
| 3 | **Height sweep.** xy fixed at r = 0.10 (in-distribution) for both object and goal; goal z ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}. | 100 BC/SAC, 50 expert |

The expert's 50 episodes are the *first 50* of the same 100 scenes, so it remains paired
with a subset of the learned arms' episodes.

### 1.8 Metrics and statistics

- **Success rate** with a **Wilson 95% score interval**, implemented directly (numpy/scipy
  only, no statsmodels dependency). Wilson is used rather than the normal approximation
  because several bins sit at or near 0% and 100%, where the normal interval is degenerate.
- **Mean and median final ‖achieved_goal − desired_goal‖ over failures only** — separates a
  near miss from a totally lost episode.
- **Episodes actually run** and **scenes rejected at generation time**, per bin.
- **McNemar's exact test** on the BC-vs-SAC discordant pairs per bin:
  `scipy.stats.binomtest(b, b + c, 0.5)`, where *b* counts scenes BC solved and SAC did not
  and *c* the reverse. This is the correct test for paired binary outcomes and is far more
  sensitive than asking whether two Wilson intervals overlap. Concordant pairs carry no
  information about which policy is better and are correctly excluded.

### 1.9 Sanity checks (run before the sweep)

1. **Positions.** Every sampled object and goal in every bin lies inside the safe rectangle,
   and object z stays at 0.020 after settling (cube never falls off the table). Minimum
   object–goal separation across all bins: 0.051 m. **PASS**
2. **Replay determinism.** `ood_scenes.npz` loaded twice gives byte-identical arrays across
   all 44 arrays, and matches the in-memory table. **PASS**
3. **Expert feasibility probe**, 10 episodes/bin: 100% in 12 of 13 bins; 80% at r = 0.30.
   Above the halt threshold, so the sweep continued. The full 50-episode run put r = 0.30 at
   92% — the only bin where the ceiling is meaningfully below 100%, and results there are
   interpreted relative to that ceiling rather than to 100%. **PASS**

The whole run was also repeated end-to-end after a plotting fix and reproduced identical
per-bin counts, confirming determinism of the pipeline.

---

## 2. Setup

All figures below were read from the machine the run executed on.

| | |
|---|---|
| OS | Windows 10 Pro 10.0.19045 (`Windows-10-10.0.19045-SP0`) |
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 threads |
| RAM | 15.9 GB |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Python | 3.13.5 (`.venv\Scripts\python.exe`) |
| torch | **2.11.0+cu130** (`torch.version.cuda = 13.0`) |
| CUDA available | `True` |
| **Device used** | **`cuda`** |
| numpy | 2.4.3 |
| scipy | 1.18.0 |
| gymnasium | 1.2.3 |
| panda_gym | 3.0.7 |
| pybullet | build Jun 21 2026 11:30:41 |

**Wall-clock:** Experiment 1 31.3 s · Experiment 2 107.8 s · Experiment 3 122.9 s ·
**total 263.1 s (4.4 min)** for 3 850 episodes.

As expected, the GPU buys nothing here. The bottleneck is PyBullet physics stepping, which
is CPU-only and single-threaded; the policy forward pass is a 3×256 MLP on a single 22-dim
observation, where kernel-launch and host↔device transfer overhead can exceed the compute.
The measured time above is on `cuda` and is reported as-is — no CPU comparison run was made,
and none is claimed.

### Commands run, verbatim

```
.venv/Scripts/python.exe -m pip install scikit-learn
.venv/Scripts/python.exe eval_ood.py --sanity
.venv/Scripts/python.exe -u eval_ood.py --skip-sanity > ood_run.log 2>&1
.venv/Scripts/python.exe eval_ood.py --replot
```

One dependency change was required. `B_cloning.py` imports `sklearn.model_selection` at
module scope and `scikit-learn` was absent from the venv, so importing `ActorNetwork` failed.
`scikit-learn 1.9.0` was installed (pulling `joblib 1.5.3`, `threadpoolctl 3.6.0`,
`narwhals 2.24.0`). It is used only by `BC.load_data`, which this evaluation never calls, so
it cannot affect any number reported here. **torch was deliberately not touched** — CUDA was
already available, and changing it mid-run would have invalidated the setup record above.

Files produced: `eval_ood.py`, `ood_scenes.npz`, `ood_results.csv`, `ood_radius.png`,
`ood_height.png`, `ood_run.log`.

---

## 3. Results

### 3.1 Experiment 1 — default distribution (in-distribution headline)

200 paired episodes per arm, seeds 20000–20205 (6 degenerate scenes skipped).

| arm | success | 95% Wilson CI | failures | mean fail dist | median fail dist |
|---|---|---|---|---|---|
| **BC** | **190/200 = 95.0%** | [0.910, 0.973] | 10 | 0.401 m | 0.274 m |
| **SAC+HER** | **104/200 = 52.0%** | [0.451, 0.588] | 96 | 0.199 m | 0.194 m |
| scripted expert | 200/200 = 100% | [0.981, 1.000] | 0 | — | — |

McNemar BC vs SAC+HER: *b* = 95, *c* = 9, **p = 3.0 × 10⁻¹⁹**. BC is better on the training
distribution by a very wide and unambiguous margin.

The expert at a clean 200/200 confirms every default-distribution scene is feasible, so
SAC's 96 failures are entirely policy failures.

Note the failure-distance asymmetry: BC's rare failures are *far* (mean 0.40 m — it drops or
flings the cube), while SAC's many failures are *close* (mean 0.20 m). SAC fails
consistently and tidily. §3.4 explains why.

### 3.2 Experiment 2 — radius sweep

![success rate vs radius](ood_radius.png)

Success rate (Wilson 95% CI). Expert n = 50, BC/SAC n = 100.

| r (m) | BC | SAC+HER | expert | McNemar *b* / *c* | *p* | rejected |
|---|---|---|---|---|---|---|
| 0.05 | **98%** [0.93, 0.99] | 38% [0.29, 0.48] | 100% | 60 / 0 | 1.7 × 10⁻¹⁸ | 15 |
| 0.10 | **96%** [0.90, 0.98] | 47% [0.38, 0.57] | 100% | 52 / 3 | 1.5 × 10⁻¹² | 9 |
| 0.15 | **86%** [0.78, 0.91] | 45% [0.36, 0.55] | 100% | 50 / 9 | 5.3 × 10⁻⁸ | 13 |
| 0.20 | **64%** [0.54, 0.73] | 35% [0.26, 0.45] | 100% | 44 / 15 | 2.0 × 10⁻⁴ | 3 |
| 0.25 | **33%** [0.25, 0.43] | 13% [0.08, 0.21] | 100% | 31 / 11 | 2.9 × 10⁻³ | 3 |
| 0.30 | 10% [0.06, 0.17] | 7% [0.03, 0.14] | 92% | 10 / 7 | 0.63 | 6 |

Failure distances (mean / median, metres, failures only):

| r (m) | BC | SAC+HER | expert |
|---|---|---|---|
| 0.05 | 0.140 / 0.140 | 0.143 / 0.148 | — |
| 0.10 | 0.101 / 0.098 | 0.178 / 0.185 | — |
| 0.15 | 0.258 / 0.202 | 0.219 / 0.212 | — |
| 0.20 | 0.394 / 0.339 | 0.307 / 0.318 | — |
| 0.25 | 0.482 / 0.426 | 0.338 / 0.354 | — |
| 0.30 | 0.487 / 0.482 | 0.422 / 0.434 | 0.377 / 0.361 |

**Reading.** BC holds ≥ 96% while fully inside the training square, begins to slip at its
edge (86% at r = 0.15), and collapses across the boundary region — 64% at r = 0.20, 33% at
r = 0.25, 10% at r = 0.30. The knee sits almost exactly where the support ends. SAC+HER
starts far lower (38–47%) and declines more gently, simply because it has much less to lose.
BC beats SAC significantly in all five bins up to r = 0.25; at r = 0.30 the two are
statistically indistinguishable (p = 0.63) because both have bottomed out.

The expert holds 100% out to r = 0.25 and 92% at r = 0.30, so degradation up to r = 0.25 is
**entirely** generalisation failure, not the workspace running out. At r = 0.30 a ~8%
feasibility penalty is present and BC's 10% should be read against a 92% ceiling.

BC's rising failure distance with radius (0.14 → 0.49 m) shows its failures shift from near
misses to gross ones: outside the support it does not merely under-shoot, it loses the cube.

**Cross-check against the prior BC-only sweep.** An earlier, cruder run (annulus sampling,
per-radius rejection, 60 eps/bin, no CIs) gave 100% / 98.3% / 98.3% / 85.0% / 41.7% / 21.7%
at r ≈ 0.025 / 0.075 / 0.125 / 0.180 / 0.234 / 0.283. The new curve (98 / 96 / 86 / 64 / 33 /
10 at r = 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.30) sits in the same ballpark and has the same
shape and knee location. The new bins are at systematically larger radii than the old ones
(e.g. 0.20 vs 0.180, 0.30 vs 0.283), and on a curve this steep that alone accounts for the
lower numbers. No gap approaches the 20-point threshold that would indicate a
scene-generation bug.

### 3.3 Experiment 3 — goal height sweep

![success rate vs goal height](ood_height.png)

xy fixed at r = 0.10 (in-distribution) for both object and goal; only goal height varies.
Training ceiling is z = 0.22, so the last three bins are OOD on an axis neither policy saw.

| goal z (m) | BC | SAC+HER | expert | McNemar *b* / *c* | *p* | rejected |
|---|---|---|---|---|---|---|
| 0.05 | 96% [0.90, 0.98] | **100%** [0.96, 1.00] | 100% | 0 / 4 | 0.125 | 16 |
| 0.10 | **98%** [0.93, 0.99] | 0% [0.00, 0.04] | 100% | 98 / 0 | 6.3 × 10⁻³⁰ | 0 |
| 0.15 | **96%** [0.90, 0.98] | 0% [0.00, 0.04] | 100% | 96 / 0 | 2.5 × 10⁻²⁹ | 0 |
| 0.20 | **100%** [0.96, 1.00] | 0% [0.00, 0.04] | 100% | 100 / 0 | 1.6 × 10⁻³⁰ | 0 |
| 0.25 | **93%** [0.86, 0.97] | 0% [0.00, 0.04] | 100% | 93 / 0 | 2.0 × 10⁻²⁸ | 0 |
| 0.30 | **72%** [0.63, 0.80] | 0% [0.00, 0.04] | 100% | 72 / 0 | 4.2 × 10⁻²² | 0 |
| 0.35 | 8% [0.04, 0.15] | 0% [0.00, 0.04] | 100% | 8 / 0 | 7.8 × 10⁻³ | 0 |

Failure distances (mean / median, metres, failures only):

| goal z (m) | BC | SAC+HER |
|---|---|---|
| 0.05 | 0.130 / 0.122 | — |
| 0.10 | 0.170 / 0.170 | 0.154 / 0.154 |
| 0.15 | 0.621 / 0.552 | 0.183 / 0.183 |
| 0.20 | — | 0.219 / 0.212 |
| 0.25 | 0.593 / 0.682 | 0.265 / 0.260 |
| 0.30 | 0.432 / 0.339 | 0.306 / 0.301 |
| 0.35 | 0.512 / 0.381 | 0.358 / 0.357 |

**Reading.** This is the sharpest result in the report. SAC+HER goes from **100%** at
z = 0.05 to **exactly 0/100** at every height from 0.10 upward — a cliff, not a decline, and
it falls *inside* the training distribution (z = 0.10, 0.15, 0.20 are all well within the
support). BC is essentially flat at 93–100% all the way to z = 0.25, i.e. it extrapolates
*past* the training ceiling of 0.22 without penalty, then degrades at 0.30 (72%) and
collapses at 0.35 (8%). The expert is 100% everywhere including z = 0.35, so every one of
these bins is physically reachable and none of the failures are feasibility artefacts.

z = 0.05 is the one bin where SAC beats BC (100% vs 96%), and the difference is not
significant (p = 0.125).

### 3.4 Why SAC+HER fails: it never lifts the cube

The 0.05 → 0.10 cliff has a single mechanical cause. A supporting diagnostic replayed 30
archived scenes per bin per arm, recording the maximum height the cube ever reached and its
xy error before and after:

| bin | arm | mean max cube z | lifted > 0.05 m | xy err before → after | success |
|---|---|---|---|---|---|
| z = 0.20 | BC | 0.208 | **100%** | 0.114 → 0.030 | 100% |
| z = 0.20 | **SAC+HER** | **0.020** | **0%** | **0.114 → 0.114** | 0% |
| z = 0.20 | expert | 0.197 | 100% | 0.114 → 0.031 | 100% |
| z = 0.10 | BC | 0.179 | 100% | 0.134 → 0.030 | 97% |
| z = 0.10 | **SAC+HER** | **0.020** | **0%** | **0.134 → 0.134** | 0% |
| z = 0.05 | BC | 0.168 | 87% | 0.120 → 0.032 | 90% |
| z = 0.05 | **SAC+HER** | **0.023** | **0%** | 0.120 → **0.027** | 100% |
| r = 0.10 | SAC+HER | 0.023 | 0% | 0.139 → 0.078 | 50% |

The cube's resting centre is z = 0.020. **SAC+HER never raises it above 0.023 in any bin** —
it has no lifting behaviour at all. What it has is a competent *pushing* behaviour: at
z = 0.05 it slides the cube from 0.120 m away to 0.027 m, scoring 100%.

This works at z = 0.05 for a purely geometric reason. With the cube left flat at z = 0.02 and
a goal at z = 0.05, the vertical gap is 0.03 m, so the 0.05 m success threshold is still
reachable provided the xy error is under √(0.05² − 0.03²) = 0.04 m. At z = 0.10 the vertical
gap alone is 0.08 m and success becomes impossible without lifting. Hence 100% → 0%.

At z ≥ 0.10 SAC does not even engage the cube: its xy error is *identical* before and after
(0.114 → 0.114, 0.134 → 0.134). Faced with an airborne goal it leaves the cube untouched
rather than pushing it into position, which is why its failure distances are so tidy and track
the goal height so closely.

This also explains the in-distribution 52%. Under the default goal distribution, 30% of goals
are forced flat and the rest are uniform in [0.02, 0.22]; the push-solvable fraction is
roughly 0.30 + 0.70 × (0.04/0.20) ≈ 0.44, and SAC scores 52% — the push-solvable subset plus
the low-goal margin. Its 52% is not "a pick-and-place policy that works half the time"; it is
**a pushing policy scoring exactly the fraction of the benchmark that pushing solves**.

---

## 4. Discussion

**The two arms are not the same kind of policy, and the aggregate number hides it.** The
headline 95% vs 52% invites the reading "BC generalises better than SAC+HER". The height
sweep shows that framing is wrong. SAC+HER did not learn a worse pick-and-place policy — it
learned a *different task*. It converged to a local optimum that solves every goal reachable
by sliding and abandons every goal that is not. Any single-number benchmark on the default
distribution would have reported this as a mediocre pick-and-place agent rather than an
excellent pusher, which is a materially different diagnosis and points at a different fix
(exploration and the lifting bottleneck, not more training on the same objective).

**BC inherited the expert's strategy, including its structure.** BC lifts in 87–100% of
episodes and its success curve tracks the expert's shape at reduced amplitude. Imitation
transferred the *whole* behaviour, and with it the generalisation profile: BC extrapolates
comfortably along the height axis it saw demonstrated (flat to z = 0.25, past the 0.22
training ceiling) and degrades along the radial axis roughly where its demonstrations stop.
The scripted expert never fails at these heights, and the demonstrations covered lifts up to
0.22, so BC extending to 0.25 is modest interpolation of a demonstrated skill rather than
novel capability. Its collapse at z = 0.35 (8%) is where the demonstrated motion no longer
reaches.

**Failure geometry distinguishes the two failure modes.** BC's failure distances grow with
distance from the support (0.14 → 0.49 m radially): it attempts the full pick-and-place and
loses the cube, so its errors are large and unstructured. SAC's failure distances stay small
and are almost entirely vertical, tracking goal height with a residual close to its xy error.
"Near miss vs totally lost" cleanly separates *a policy attempting the right task and failing*
from *a policy executing a different task correctly*.

**The radial knee lines up with the support boundary.** BC sits at 96–98% inside the square,
86% at its edge, and drops steeply through r = 0.212 where the square is fully exited. That
the knee coincides with the geometric boundary — with the expert still at 100% through
r = 0.25 — is the cleanest available evidence that this is distributional, not kinematic.

**What the pairing bought.** McNemar's test resolves differences the CI-overlap heuristic
would miss. At r = 0.25 the Wilson intervals (BC [0.25, 0.43], SAC [0.08, 0.21]) already fail
to overlap, but at r = 0.20 (BC [0.54, 0.73], SAC [0.26, 0.45]) and at several other bins,
reading significance off interval overlap is both conservative and wrong in principle. With
paired scenes, *b* and *c* count the scenes that actually discriminate the two policies —
44 vs 15 at r = 0.20, p = 2 × 10⁻⁴ — which is a far stronger statement from the same data.

---

## 5. Limitations

1. **n = 1 training seed per arm.** Each arm is a single training run, so every between-arm
   difference confounds *method* with *training-run variance*. This is the most serious
   limitation here. Nothing in this report can distinguish "SAC+HER converges to pushing" from
   "this particular SAC+HER run converged to pushing"; the latter is the more defensible
   claim, and separating them needs several seeds per method.
2. **Different checkpoint-selection criteria.** BC's `95_best` was selected on held-out
   behavioural-cloning MSE; SAC's `291904_interrupted` was taken from a training-curve
   plateau at an interrupted run. These are different criteria optimising different
   quantities, which biases the comparison by an unknown amount. Neither criterion touched
   the evaluation seeds, so the comparison is not contaminated — but it is not a like-for-like
   model-selection protocol either.
3. **The +x wedge is excluded from all swept bins.** The 96.4° arc around +x is never sampled
   in Experiments 2 or 3, so those bins do not represent the training support uniformly, and
   the in-distribution points of the sweeps are not directly comparable to Experiment 1. This
   was a deliberate trade to keep the angular distribution constant across radii; Experiment 1
   covers the full square as a check, and its BC/SAC numbers (95% / 52%) are consistent with
   the small-radius sweep bins.
4. **Resolution.** 100 episodes/bin gives a Wilson half-width of roughly ±10 points near 50%
   success (e.g. r = 0.20 BC: 64%, [0.54, 0.73]). Differences smaller than that are not
   resolvable per-bin, though McNemar on paired outcomes is considerably more sensitive than
   that half-width implies. The expert's 50 episodes/bin are coarser still (±~8 points at
   92%).
5. **The height sweep fixes xy at a single radius.** Experiment 3 holds r = 0.10 for both
   object and goal, so it measures the height axis at one radial slice only; height–radius
   interaction is unmeasured.
6. **Simulation-specific and build-specific.** All results are PyBullet with this build
   (Jun 21 2026). PyBullet is deterministic for a given build and inputs, but trajectories can
   differ slightly across builds and platforms. Internal validity is unaffected — all three
   arms ran on the same machine from the same scene table — but absolute numbers may shift
   elsewhere.
7. **The lift diagnostic in §3.4 uses 30 episodes per bin**, not the full 100, and is
   supporting evidence rather than a primary result. Its central claim (zero lifts by SAC) is
   nonetheless unambiguous within that sample: across 120 instrumented SAC episodes the
   maximum cube height never exceeded 0.023 m, against a 0.020 m resting height. The full
   sweep did not record cube height, so "never lifts" is established on the diagnostic
   subset and is consistent with — but not directly proven by — the 0/100 results at every
   z ≥ 0.10.
