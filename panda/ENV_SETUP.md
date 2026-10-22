# Panda 训练工程环境配置与复现说明 (T01—T05)

- **创建时间**：2026-10-20
- **对应任务**：T01（Lift 复现）、T02（运动学/视觉定位）、T03（PickPlace）、T04（示范与 BC）、T05（PPO 强化学习）
- **隔离原则**：独立 Conda 环境 `panda`，严禁与 `tiago_sim` 或其他项目混装依赖。

---

## 1. 宿主与运行环境基线

- **操作系统**：Windows 11 家庭版 + WSL2 (Ubuntu 22.04.5 LTS)
- **Linux 内核**：`5.15.167.4-microsoft-standard-WSL2`
- **显卡/显存**：NVIDIA GeForce RTX 3060 Laptop GPU (6144 MiB)
- **显卡驱动**：527.58 (WSL Direct3D / CUDA 直通支持)
- **图形后端**：WSLg 原生支持 (`DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`)
  - 无头高性能物理/视觉计算：`MUJOCO_GL=egl`
  - 交互式窗口可视化：`MUJOCO_GL=glfw`

---

## 2. 核心已安装软件包与版本清单

本环境基于 **Python 3.10.22** 构建，核心包精确版本如下：

| 软件包 | 安装版本 | 功能职责与作用 |
|:---|:---|:---|
| **robosuite** | `1.4.1` | 机械臂操作仿真环境标准库（Panda 模型与 Lift/PickPlace 任务） |
| **mujoco** | `2.3.7` | DeepMind 官方物理仿真引擎（**必须 2.x**，与 robosuite 1.4.1 配套） |
| **numpy** | `2.2.6` | 科学计算底层数组（与 robosuite 1.4.1/PyPI 兼容对齐） |
| **scipy** | `1.15.3` | 空间变换、旋转矩阵与插值算法库（运动学与轨迹规划） |
| **numba** | `0.68.0` | 机械臂动力学与碰撞检测 JIT 高性能加速 |
| **llvmlite** | `0.50.0` | numba 的 LLVM 底层代码生成库 |
| **opencv-python**| `5.0.0.93` | RGB-D 图像处理、深度反投影与视觉感知 |
| **Pillow** | `12.3.0` | 相机观察图像转换与录像帧保存 |
| **glfw** | `2.11.0` | 本地 OpenGL 图形窗口管理与键盘/鼠标交互 |
| **pynput** | `1.8.2` | 键盘遥操作控制输入驱动 |

> 完整锁定清单见同一目录下的 `requirements.lock`。

---


---

## 3.1 版本兼容性重要结论（2026-10-20 实测）

- **robosuite 1.4.1 必须搭配 mujoco 2.x，不能装 mujoco 3.x。**
- robosuite 1.4.1 的元数据只写了 `mujoco>=2.3.0`，pip 在 2026 年会把 mujoco 3.16.0 解析进来；
- mujoco 3.x 移除了 `MjData.qM` 并重写了枚举绑定（`mjtJoint` 为 pybind11 enum），导致两类报错：
  - `binding_utils.py: assert joint_type in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)` → AssertionError
  - `base_controller.py: mujoco.mj_fullM(..., self.sim.data.qM)` → AttributeError: no attribute 'qM'
- **已修复方式**：卸载 mujoco 3.16.0，改用 mujoco 2.3.7（wheel 从 PyPI 清华镜像下载，本地安装）。
- 修复后官方 robosuite 入门脚本（Panda + Lift，1000 步随机动作 + render）可完整运行，退出码 0。
- 如未来需要 mujoco 3.x（T05 PPO 或新版教程），应同步升级 robosuite 至 1.5.1+（官方要求 mujoco>=3.2.3），不要混搭。


---

## 3.2 渲染配置（2026-10-22 重建环境时修正）

**问题**：robosuite 1.4.1 的 `macros_private.py` 默认 `MUJOCO_GPU_RENDERING = True`，
`binding_utils.py:37` 会在 Linux 下**强制把 `MUJOCO_GL` 改成 `egl`**，无视命令行设置。
本机 EGL 设备初始化失败，导致 `has_renderer=True` 报
`Cannot initialize a EGL device display`。

**修正**：编辑

```
~/miniconda3/envs/panda/lib/python3.10/site-packages/robosuite/macros_private.py
```

把 `MUJOCO_GPU_RENDERING = True` 改为 `False`。

**注意**：这是对 site-packages 的修改，重装 robosuite 后需重做一次。

**修正后的渲染能力**（WSLg + Mesa D3D12）：

```
OpenGL renderer: D3D12 (NVIDIA GeForce RTX 3060 Laptop GPU)
OpenGL version : 4.2 (Compatibility Profile) Mesa 23.2.1
direct rendering: Yes
```

两种模式均验证通过：

| 模式 | 命令 | 用途 |
|---|---|---|
| 图形窗口 | `MUJOCO_GL=glfw` | 组会演示、录屏 |
| 无头渲染 | `MUJOCO_GL=egl` | 批量跑数据 |

WSLg 所需系统库（cloud image 需手动装）：

```bash
sudo apt install -y x11-utils x11-apps mesa-utils libglu1-mesa \
  libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers \
  libx11-6 libxext6 libxrender1 libxi6 libxcursor1 libxrandr2 libxinerama1 \
  fonts-dejavu-core
```

---

## 3.3 系统库依赖（cloud image 必装）

minimal cloud image 缺少 opencv 所需的系统库，否则 `import robosuite` 会报
`ImportError: libGL.so.1`：

```bash
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libegl-mesa0
```

## 3. T01 自主冒烟测试指引（由学生自主执行）

在终端进入 WSL，激活环境或直接指定环境解释器执行：

### 模式 A：无头高性能 EGL 冒烟验证（无需弹窗，验证物理+动作循环）
```bash
MUJOCO_GL=egl ~/miniconda3/envs/panda/bin/python -c "
import robosuite as suite
import numpy as np

# 实例化 Panda + Lift 任务环境
env = suite.make('Lift', robots='Panda', has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
obs = env.reset()
print('Panda Lift 环境实例化成功！动作空间维度:', env.action_dim)

# 执行 20 步随机动作循环
for step in range(20):
    action = np.random.uniform(-1, 1, env.action_dim)
    obs, reward, done, info = env.step(action)
print('20 步动作步进循环顺利执行，当前 reward:', reward)
env.close()
print('【无头冒烟测试通过】')
"
```

### 模式 B：GLFW 桌面交互窗口验证（弹窗查看真实机械臂随机动作）
```bash
MUJOCO_GL=glfw ~/miniconda3/envs/panda/bin/python -m robosuite.demos.demo_random_action --environment Lift --robots Panda
```
*注：该命令会在 WSLg 中弹出一个 3D 可视化窗口，Panda 机械臂在桌面上做随机控制动作。按键盘 `ESC` 或关闭窗口即可退出。*

---

## 4. 组会展示与边界须知

1. **“安装成功” ≠ “任务完成”**：`demo_random_action` 仅代表安装与渲染管线打通，Lift 任务需要结合控制器实现真正的闭环取放。
2. **状态/动作接口位置**：在 robosuite 中，观测变量集中在 `obs` 字典（如 `robot0_eef_pos`, `cube_pos` 等），动作空间 `env.action_dim` 对应末端位姿控制量。
