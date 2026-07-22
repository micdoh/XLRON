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
RUNS = {
    # run_id: (label, color)
    "cnzbizzb": ("SHAC v1.0 flat, T=5", "#8A7DDF"),
    "shac_sc_t1_lr3e4": ("SHAC v1.1 slot-cond, T=1", "#3E7DDB"),
    "shac_sc_t5_lr3e4": ("SHAC v1.1 slot-cond, T=5", "#2FA396"),
    "shac_hyb_t1_lr3e4": ("SHAC v1.2 hybrid (+PG), T=1", "#D96D4F"),
    "shac_pg_only_t1": ("PG-only ablation, T=1", "#B58A2A"),
    "ppo_ref_nsfnet": ("PPO reference", "#C065A8"),
}
BASELINE_BP = 0.026  # KSP-FF at load 250 (run t0hxabuu)
METRIC = "service_blocking_probability_mean"

api = wandb.Api()
project_runs = {r.id: r for r in api.runs("micdoh/DIFF_SIM_RL")}
by_name = {}
for r in project_runs.values():
    by_name.setdefault(r.name, r)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

for key, (label, color) in RUNS.items():
    run = project_runs.get(key) or by_name.get(key)
    if run is None:
        continue
    hist = run.history(keys=[METRIC, "env_step"], pandas=True, samples=2000)
    if hist.empty or METRIC not in hist:
        continue
    hist = hist.dropna(subset=[METRIC]).sort_values("env_step")
    ax.plot(
        hist["env_step"] / 1e6,
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
