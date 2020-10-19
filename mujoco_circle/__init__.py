from gym.envs.registration import register

register(
    id='AntCircle-v0',
    entry_point='mujoco_circle.envs:AntCircleEnv',
)
register(
    id='HumanoidCircle-v0',
    entry_point='mujoco_circle.envs:HumanoidCircleEnv',
)