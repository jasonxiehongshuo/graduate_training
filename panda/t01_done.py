import numpy as np
import robosuite as suite

env = suite.make("Lift", robots="Panda", has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False,
                 horizon=30, hard_reset=False)
#重置环境
env.reset()
for t in range(35):
    obs, r, d, info = env.step(np.zeros(env.action_dim))
    if env.timestep == 1:
        print(f"1 个动作 = {int(env.control_timestep/env.model_timestep)} 次物理积分 "
              f"(control_timestep={env.control_timestep}s, model_timestep={env.model_timestep}s)")
    if d:
        print(f"done=True 于第 {t+1} 个动作, timestep={env.timestep}, horizon={env.horizon}, cur_time={env.cur_time:.2f}s")
        break
try:
    env.step(np.zeros(env.action_dim))
except ValueError as e:
    print(f"done 后 step -> ValueError: {e}")
env.close()
