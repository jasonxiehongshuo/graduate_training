import numpy as np
import robosuite as suite

BASE = dict(robots="Panda", 
            has_renderer=False, 
            has_offscreen_renderer=False,
            use_camera_obs=False, 
            horizon=60, 
            hard_reset=False,
            initialization_noise=None)

ACTION = np.array([0, 0, 1, 0, 0, 0, 0, 0])   # 纯 +z 方向

print(f"{'freq':>6s} {'每步仿真时长':>14s} {'20步总时长':>12s} {'dz(m)':>10s} {'最终 cur_time':>14s}")
for freq in (10, 20, 40):
    env = suite.make("Lift", **BASE, control_freq=freq)
    #env.reset() #返回一个obs字典
    z0 = float(np.array(env.reset()["robot0_eef_pos"])[2])
    # 取出obs["robot0_eef_pos"]的z坐标，作为初始高度
    for _ in range(20): # 执行20步action，记录每步的仿真时长和最终的cur_time
        obs, _, _, _ = env.step(ACTION)
    # 取出obs["robot0_eef_pos"]的z坐标，作为最终高度
    z1 = float(np.array(obs["robot0_eef_pos"])[2])
    print(f"{freq:>6d} {env.control_timestep:>14.4f} {20*env.control_timestep:>12.2f} "
          f"{z1 - z0:>10.4f} {env.cur_time:>14.2f}")
    # cur_time是系统时间，表示从环境开始运行到当前的时间，单位为秒
    env.close()