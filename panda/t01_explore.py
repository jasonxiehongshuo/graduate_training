import os
import sys

import numpy as np
import robosuite as suite

OUT = "/home/xie/graduate_training/results/raw/task01"
os.makedirs(OUT, exist_ok=True)


def build_env(horizon=200, reward_shaping=False, placement=None):
    return suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=reward_shaping,
        reward_scale=1.0,
        control_freq=20,
        horizon=horizon,
        hard_reset=True,
        placement_initializer=placement,
    )


def fixed_cube_env(x=0.0, y=0.0, z=0.02):
    """把 cube 固定在指定位置，用于对照实验（目标位置修改）。"""
    from robosuite.utils.placement_samplers import UniformRandomSampler

    env = build_env(horizon=200)
    sampler = UniformRandomSampler(
        name="FixedSampler",
        mujoco_objects=env.cube,
        x_range=[x, x],
        y_range=[y, y],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=env.table_offset,
        z_offset=z,
    )
    env.close()
    return build_env(horizon=200, placement=sampler)


def run_episode(env, seed=0):
    np.random.seed(seed)
    obs = env.reset()
    cube_id = env.cube_body_id
    table_z = env.model.mujoco_arena.table_offset[2]
    threshold = table_z + 0.04
    rec = {"t": [], "cube_z": [], "eef_z": [], "reward": [], "done": []}
    for i in range(env.horizon):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)
        rec["t"].append(env.cur_time)  # 累计仿真时间(秒)，非步数累乘
        rec["cube_z"].append(float(env.sim.data.body_xpos[cube_id][2]))
        rec["eef_z"].append(float(obs["robot0_eef_pos"][2]))
        rec["reward"].append(float(reward))
        rec["done"].append(bool(done))
        if done:
            break
    success = max(rec["cube_z"]) > threshold
    return rec, success, threshold


def mode_inspect():
    env = build_env()
    print("=" * 64)
    print("T01 inspect: Panda + Lift 接口解剖")
    print("=" * 64)
    print("环境类型        :", type(env).__name__)
    print("动作维度        :", env.action_dim)
    print("动作下限 action_spec[0]:", env.action_spec[0])
    print("动作上限 action_spec[1]:", env.action_spec[1])
    print("horizon         :", env.horizon)
    print("control_freq    :", env.control_freq, "Hz  物理步长:", env.timestep, "s")

    obs = env.reset()
    print("\n--- reset 返回的 observation keys ---")
    for k, v in obs.items():
        a = np.asarray(v)
        print(f"  {k:28s} shape={a.shape} dtype={a.dtype}")
    print("\n--- 成功判据（源码 lift.py _check_success）---")
    table_z = env.model.mujoco_arena.table_offset[2]
    print(f"  桌面高度 = {table_z}, 判据 cube z > {table_z + 0.04:.2f} 视为 Lift 成功")
    print(f"  reset 后 cube z = {env.sim.data.body_xpos[env.cube_body_id][2]:.4f}")

    rec, success, threshold = run_episode(env, seed=0)
    np.savez(
        os.path.join(OUT, "random_episode.npz"),
        **{k: np.array(v) for k, v in rec.items()},
    )
    print(f"\n--- 随机策略 1 回合（{len(rec['t'])} 步）---")
    print(f"  max cube_z = {max(rec['cube_z']):.4f}  (判据 {threshold:.2f})")
    print(f"  reward 累计 = {sum(rec['reward']):.4f}")
    print(f"  done 出现   = {any(rec['done'])}")
    print("  数据保存    :", os.path.join(OUT, "random_episode.npz"))
    env.close()


def mode_variant():
    print("=" * 64)
    print("T01 variant: 三种配置对照（各 5 回合随机策略）")
    print("=" * 64)
    variants = {
        "baseline": {"reward_shaping": False, "placement": None, "note": "默认稀疏奖励 + 方块随机初始位置"},
        "dense": {"reward_shaping": True, "placement": None, "note": "稠密奖励（控制参数修改）"},
        "fixed_cube": {"reward_shaping": False, "placement": "fixed", "note": "方块固定初始位置（目标位置修改）"},
    }
    rows = []
    for name, cfg in variants.items():
        if cfg["placement"] == "fixed":
            env = fixed_cube_env(x=0.0, y=0.0, z=0.02)
        else:
            env = build_env(horizon=200, reward_shaping=cfg["reward_shaping"])
        success_count = 0
        reward_sums = []
        for seed in range(5):
            rec, success, threshold = run_episode(env, seed=seed)
            success_count += int(success)
            reward_sums.append(sum(rec["reward"]))
        rows.append((name, cfg["note"], success_count, np.mean(reward_sums), np.std(reward_sums)))
        env.close()

    print(f"\n{'变体':10s} {'成功率(5回合)':14s} {'平均累计reward':16s} {'std':8s}  说明")
    for name, note, sc, rm, rs in rows:
        print(f"{name:10s} {sc}/5            {rm:8.4f}        {rs:8.4f}  {note}")
    print("\n观察点：")
    print("  - dense 变体的 reward 是否更平滑（每一步都有反馈）")
    print("  - fixed_cube 变体的方块初始位置固定后，随机动作成功率是否变化")
    print("  - 随机策略本身基本不可能 Lift 成功，这正是后续控制器/BC/PPO 要解决的目标")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": mode_inspect, "variant": mode_variant}[mode]()