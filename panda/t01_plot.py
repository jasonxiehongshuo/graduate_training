"""T01 状态/动作曲线：cube/eef 高度、奖励、终止标志、动作幅度。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NPZ = "/home/xie/graduate_training/results/raw/task01/random_episode.npz"
OUT = "/home/xie/graduate_training/results/figures/task01/random_episode.png"
THR = 0.84

d = np.load(NPZ)
t = d["t"]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax = axes[0]
ax.plot(t, d["cube_z"], label="cube z", color="#c0392b", lw=2.0)
ax.plot(t, d["eef_z"], label="eef z", color="#2980b9", lw=1.3, alpha=0.85)
ax.axhline(THR, color="k", ls="--", lw=1.2, label=f"success threshold {THR}")
ax.axhline(0.80, color="gray", ls=":", lw=0.9, label="table top 0.80")
ax.set_ylabel("height (m)")
ax.set_ylim(0.78, 1.05)
ax.set_title("T01 Lift | random policy: cube never lifted above table (max 0.8202 m < 0.84)")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.plot(t, d["reward"], color="#27ae60", lw=1.6)
ax.set_ylabel("reward")
ax.set_ylim(-0.05, 1.05)
ax.set_title("sparse reward stays exactly 0 for all 200 steps", fontsize=9)

ax = axes[2]
mag = np.linalg.norm(d["action"], axis=1)
ax.plot(t, mag, color="#8e44ad", lw=1.0, alpha=0.8, label="|action| (8-dim norm)")
ax.set_ylabel("action magnitude")
ax.set_xlabel("simulation time (s)")
ax.set_title("random action magnitude: no structure, no goal", fontsize=9)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print("saved:", OUT)