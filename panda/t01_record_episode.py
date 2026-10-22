"""重跑一次随机回合，记录正确的仿真时间与动作，用于曲线图。"""
import os
import numpy as np
import robosuite as suite

OUT = "/home/xie/graduate_training/results/raw/task01"
os.makedirs(OUT, exist_ok=True)

env = suite.make("Lift", robots="Panda", has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 control_freq=20, horizon=200, hard_reset=False,
                 initialization_noise=None)

np.random.seed(0)
obs = env.reset()
cube_id = env.cube_body_id
table_z = env.model.mujoco_arena.table_offset[2]
thr = table_z + 0.04

rec = {k: [] for k in ["t", "cube_z", "eef_z", "reward", "done", "action"]}
for _ in range(env.horizon):
    a = np.random.uniform(-1, 1, env.action_dim)
    obs, r, d, _ = env.step(a)
    rec["t"].append(env.cur_time)
    rec["cube_z"].append(float(env.sim.data.body_xpos[cube_id][2]))
    rec["eef_z"].append(float(obs["robot0_eef_pos"][2]))
    rec["reward"].append(float(r))
    rec["done"].append(bool(d))
    rec["action"].append(a)
    if d:
        break

np.savez(os.path.join(OUT, "random_episode.npz"),
         **{k: np.array(v) for k, v in rec.items()})

t = np.array(rec["t"])
print("steps =", len(t))
print("t range = %.2f .. %.2f s" % (t.min(), t.max()))
print("cube_z init=%.4f final=%.4f max=%.4f  threshold=%.2f" %
      (rec["cube_z"][0], rec["cube_z"][-1], max(rec["cube_z"]), thr))
print("eef_z  init=%.4f final=%.4f" % (rec["eef_z"][0], rec["eef_z"][-1]))
print("reward sum=%.4f nonzero=%d" % (sum(rec["reward"]), sum(1 for x in rec["reward"] if x != 0)))
print("done at t=%.2f" % t[-1])
env.close()