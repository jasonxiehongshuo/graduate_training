import time
import numpy as np
import robosuite as suite

BASE = dict(robots="Panda", has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, horizon=20)

# 1 基本行为
env = suite.make("Lift", **BASE, hard_reset=False)
print("=== 1. reset 基本行为 (hard_reset=False) ===")
for i in range(3):
    obs = env.reset()
    # 记录 cube 和 eef(末端执行器) 的位置，timestep，cur_time，done
    print(f"  #{i+1} cube_pos={np.array(obs['cube_pos']).round(4)} "
          f"eef={np.array(obs['robot0_eef_pos']).round(4)} "
          f"timestep={env.timestep} cur_time={env.cur_time} done={env.done}")

# 2 hard_reset 差异
print("=== 2. hard_reset True vs False ===")
for hr in (False, True):
    env.close()
    env = suite.make("Lift", **BASE, hard_reset=hr)
    t0 = time.time(); 
    env.reset(); 
    dt1 = time.time() - t0
    sid = id(env.sim)
    env.step(np.zeros(env.action_dim))
    t0 = time.time(); 
    env.reset(); 
    dt2 = time.time() - t0
    print(f"  hard_reset={hr}: 首次 reset {dt1*1000:.1f}ms, 二次 {dt2*1000:.1f}ms, sim id 变化={sid != id(env.sim)}")

# 3 seed 复现（同一个随机种子，反复重置环境，方块的位置是不是每次都一样？）
print("=== 3. seed 与可复现性 ===")
env.close()
env = suite.make("Lift", **BASE, hard_reset=False)
for seed in (0, 0, 1, 1):
    np.random.seed(seed)
    obs = env.reset()
    print(f"  seed={seed}: cube_pos={np.array(obs['cube_pos']).round(4)}")

# 4 deterministic_reset
print("=== 4. deterministic_reset=True ===") #让环境在每次重置时，不再随机摆放物体或扰动机器人，而是回到一个固定的、可预测的初始状态
env.close()
env = suite.make("Lift", **BASE, hard_reset=False)
env.deterministic_reset=True
ps = [np.array(env.reset()["cube_pos"]) for _ in range(3)]
for i, p in enumerate(ps):
    print(f"  #{i+1} cube_pos={p.round(4)}")
print(f"  标准差={np.array(ps).std(0).round(6)}")
env.close()