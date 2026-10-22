import numpy as np, robosuite as suite

for shaping in (False, True):
    env = suite.make("Lift", robots="Panda", 
                     has_renderer=False, #实时可视化窗口
                     has_offscreen_renderer=False, #离屏渲染（不显示窗口）用于生成图像数据
                     use_camera_obs=False,#是否使用相机观测
                     horizon=100, hard_reset=False, #最大步数，是否硬重置
                     reward_shaping=shaping) #启用奖励塑形
    
    np.random.seed(0);# 设置随机种子，确保每次运行结果一致
    env.reset()# 重置环境，返回初始观测
    rs = []
    for _ in range(100):
        #随机动作，范围[-1, 1]，维度为动作空间维度
        obs, r, d, _ = env.step(np.random.uniform(-1, 1, env.action_dim))
        rs.append(r)
        if d: break
    # 过滤掉奖励为零的步数
    nz = [x for x in rs if x != 0]

    print(f"shaping={shaping}: 步数={len(rs)} 非零={len(nz)} "
          f"累计={sum(rs):.4f} 非零范围=[{min(nz) if nz else 0:.4f}, {max(nz) if nz else 0:.4f}]")
    env.close()