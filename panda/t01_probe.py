import robosuite as suite

env = suite.make("Lift", robots="Panda", has_renderer=False,
                 has_offscreen_renderer=False, use_camera_obs=False)
print("controller:", env.robots[0].controller_config["type"])
print("关节 dof:", env.robots[0].dof, "夹爪 dof:", env.robots[0].gripper.dof)
print("action_dim:", env.action_dim, "spec:", env.action_spec)
env.close()
