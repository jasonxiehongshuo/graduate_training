import numpy as np
import robosuite as suite
# 导入 UniformRandomSampler 用于在指定范围内随机采样物体位置
from robosuite.utils.placement_samplers import UniformRandomSampler
# 修改目标位置的范围，确保物体在指定位置附近随机摆放
# horizon 一个回合允许多少步
BASE = dict(robots="Panda", has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, horizon=50, hard_reset=False,
            initialization_noise=None)


def make_env(fixed=None):
    env = suite.make("Lift", **BASE)
    if fixed is None:
        return env
    # 创建一个 UniformRandomSampler，指定物体位置的范围为固定位置
    sampler = UniformRandomSampler(
        name="FixedSampler", mujoco_objects=env.cube,
        x_range=[fixed[0], fixed[0]], y_range=[fixed[1], fixed[1]],
        rotation=None, ensure_object_boundary_in_range=False,
        # 确保物体放置在有效范围内
        ensure_valid_placement=True,
        reference_pos=env.table_offset, z_offset=0.01)
    env.close()
    # 重新创建环境，并使用自定义的采样器来固定物体位置
    return suite.make("Lift", **BASE, placement_initializer=sampler)


groups = [("A0 随机基线", None),
          ("A1 固定(0,0)", (0.0, 0.0)),
          ("A2 固定(0.05,0)", (0.05, 0.0))]

print(f"{'组别':16s} {'cube_pos 均值':28s} {'标准差':24s} eef 标准差")
for label, fixed in groups:
    env = make_env(fixed)
    ps, es = [], []
    for _ in range(5):
        obs = env.reset()
        ps.append(np.array(obs["cube_pos"]))
        es.append(np.array(obs["robot0_eef_pos"]))
    env.close()
    ps, es = np.array(ps), np.array(es)
    print(f"{label:16s} {str(ps.mean(0).round(4)):28s} {str(ps.std(0).round(5)):24s} {es.std(0).round(5)}")

