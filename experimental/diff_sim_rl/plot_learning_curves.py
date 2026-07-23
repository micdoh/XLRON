"""Pull DIFF_SIM_RL learning curves from wandb and render the Phase A figure.

Usage: uv run python experimental/diff_sim_rl/plot_learning_curves.py
Writes figures/phase_a_nsfnet_curves.png (PNG per project preference).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# Fixed identity -> color mapping (colors follow the entity, never the rank).
# Hues from a CVD-safe categorical set; baseline is neutral ink.
# run name (or id): (label, color, num_envs). wandb's env_step counts logged
# per-rollout-step points, so true env steps = env_step * NUM_ENVS.
RUNS = {
    "cnzbizzb": ("SHAC flat T=5 (best analytic)", "#3E7DDB", 256),
    "shac_flat_t5_seed2": ("SHAC flat T=5, seed 2", "#8FB3E8", 256),
    "shac_flat_t5_nostate": ("no cross-step grads (ablation)", "#2FA396", 256),
    "shac_sc_t1_lr3e4": ("SHAC slot-cond T=1", "#8A7DDF", 256),
    "shac_sc_t5_lr3e4": ("SHAC slot-cond T=5", "#B58A2A", 256),
    "shac_pg_only_t1": ("REINFORCE-only (diverged)", "#D96D4F", 256),
    "ppo_ref_nsfnet": ("PPO reference", "#C065A8", 64),
    "shac_il_h32": ("PPO + analytic interleave", "#4C4C55", 256),
}
BASELINE_BP = 0.026  # KSP-FF at load 250 (run t0hxabuu)
METRIC = "service_blocking_probability_mean"

api = wandb.Api()
project_runs = {r.id: r for r in api.runs("micdoh/DIFF_SIM_RL")}
by_name = {}
for r in project_runs.values():
    by_name.setdefault(r.name, r)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

for key, (label, color, n_envs) in RUNS.items():
    run = project_runs.get(key) or by_name.get(key)
    if run is None:
        continue
    hist = run.history(keys=[METRIC, "env_step"], pandas=True, samples=2000)
    if hist.empty or METRIC not in hist:
        continue
    hist = hist.dropna(subset=[METRIC]).sort_values("env_step")
    ax.plot(
        hist["env_step"] * n_envs / 1e6,
        hist[METRIC] * 100,
        label=label,
        color=color,
        linewidth=1.6,
    )

ax.axhline(BASELINE_BP * 100, color="#555555", linewidth=1.2, linestyle="--")
ax.annotate(
    "KSP-FF 2.60%",
    xy=(0.99, BASELINE_BP * 100),
    xycoords=("axes fraction", "data"),
    ha="right",
    va="bottom",
    fontsize=8,
    color="#555555",
)

ax.set_xlabel("Environment steps (millions)")
ax.set_ylabel("Service blocking probability (%)")
ax.set_title("Phase A: SHAC on NSFNET (DeepRMSA setting, load 250)", fontsize=10)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False)

out = os.path.join(os.path.dirname(__file__), "figures", "phase_a_nsfnet_curves.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print(f"saved {out}")

# ---- Phase A2: distribution actions + TV shaping ----
RUNS_A2 = {
    "cnzbizzb": ("flat surrogate (control, 4.3%)", "#3E7DDB", 256),
    "shac_flat_tv05": ("flat + TV shaping", "#8FB3E8", 256),
    "shac_dist_tv05": ("dist + TV shaping", "#D96D4F", 256),
    "shac_tfm_dist_tv": ("transformer + dist + TV", "#2FA396", 64),
    "shac_tfm_dist_tv_lr15": ("transformer + dist + TV, LR 1.5e-3", "#B58A2A", 64),
}
fig2, ax2 = plt.subplots(figsize=(8, 4.8), dpi=150)
for key, (label, color, n_envs) in RUNS_A2.items():
    run = project_runs.get(key) or by_name.get(key)
    if run is None:
        continue
    hist = run.history(keys=[METRIC, "env_step"], pandas=True, samples=2000)
    if hist.empty or METRIC not in hist:
        continue
    hist = hist.dropna(subset=[METRIC]).sort_values("env_step")
    ax2.plot(hist["env_step"] * n_envs / 1e6, hist[METRIC] * 100,
             label=label, color=color, linewidth=1.6)
ax2.axhline(BASELINE_BP * 100, color="#555555", linewidth=1.2, linestyle="--")
ax2.annotate("KSP-FF 2.60%", xy=(0.99, BASELINE_BP * 100),
             xycoords=("axes fraction", "data"), ha="right", va="bottom",
             fontsize=8, color="#555555")
ax2.set_xlabel("Environment steps (millions)")
ax2.set_ylabel("Service blocking probability (%)")
ax2.set_title("Phase A2: distribution actions + TV shaping (NSFNET, load 250)", fontsize=10)
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(fontsize=8, frameon=False)
out2 = os.path.join(os.path.dirname(__file__), "figures", "phase_a2_nsfnet_curves.png")
fig2.tight_layout()
fig2.savefig(out2)
print(f"saved {out2}")
