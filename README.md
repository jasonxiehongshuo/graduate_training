# CC-Robot-AI-Lab 研究生自主项目训练 V5.1 索引

## 1. 运行环境与工程划分

本项目遵循《手册 V5.1》“一个引擎，三个相互独立的训练工程”原则：
- **宿主环境**：WSL2 (Ubuntu 22.04.5 LTS), Python 3.10+, NVIDIA RTX 3060 Laptop (6GB), WSLg 窗口 + EGL 无头渲染双支持
- **`panda/`**：Panda 机械臂操作、示范采集与强化学习（T01—T05），独立 Conda 环境 `panda`（Python 3.10, robosuite 1.4.1, MuJoCo）
- **`stretch/`**：Stretch 3 移动抓取与导航（T06—T10），stretch_mujoco 独立环境
- **`tiago/`**：TIAGo++ 移动双臂迁移（T11），Conda 环境 `tiago_sim`（MuJoCo 3.3.7, Menagerie pal_tiago_dual）
- **`results/`**：原始数据（raw）、曲线图表（figures）、运行录像（videos），按 `task01`—`task12` 分任务严密归档

---

## 2. 12 项任务执行索引

| 任务号 | 平台 | 任务内容 | 状态 | 关键提交 / 版本 | 原始数据与结果路径 | 组会展示核心结论 |
|:---|:---|:---|:---:|:---|:---|:---|
| **T01** | Panda | 复现 Lift 基础操作任务 | 进行中 | robosuite 1.4.1 | `results/raw/task01/` | 环境打通与最小冒烟测试 |
| **T02** | Panda | 运动学、规划与 RGB-D 三维定位 | 待启动 | — | `results/raw/task02/` | — |
| **T03** | Panda | PickPlace 抓取与放置基线 | 待启动 | — | `results/raw/task03/` | — |
| **T04** | Panda | 示范数据采集、行为克隆 (BC) | 待启动 | — | `results/raw/task04/` | — |
| **T05** | Panda | PPO 强化学习与基线公平对照 | 待启动 | — | `results/raw/task05/` | — |
| **T06** | Stretch | 移动操作机器人及传感接口 | 待启动 | — | `results/raw/task06/` | — |
| **T07** | Stretch | 自主导航与避障 (原生/Nav2) | 待启动 | — | `results/raw/task07/` | — |
| **T08** | Stretch | 工作位选择与可达性分析 | 待启动 | — | `results/raw/task08/` | — |
| **T09** | Stretch | RGB-D 目标定位与坐标转换 | 待启动 | — | `results/raw/task09/` | — |
| **T10** | Stretch | 视觉引导导航—抓取—放置全链路 | 待启动 | — | `results/raw/task10/` | — |
| **T11** | TIAGo++| 双臂操作复现与移动双臂迁移 | 部分就绪 | MuJoCo 3.3.7 (Menagerie) | `results/raw/task11/` | pal_tiago_dual 双臂+底盘模型验证通过 |
| **T12** | 研究 | 独立科研预研：可证伪假设与对照闭环 | 待启动 | — | `results/raw/task12/` | — |

---

# T01 TASK

1.环境安装后测试
``` python
        MUJOCO_GL=glfw python3 -m robosuite.demos.demo_device_control
```
> 如果不想每次敲MUJOCO_GL=glfw
> 需要配置环境变量，具体配置如下
``` python
        mkdir -p ~/miniconda3/envs/panda/etc/conda/activate.d
        echo 'export MUJOCO_GL=glfw' > ~/miniconda3/envs/panda/etc/conda/activate.d/mujoco_gl.sh
```



