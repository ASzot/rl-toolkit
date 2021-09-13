import rlf.envs.ant.ant_interface
from gym.envs.registration import register

register(
    id="AntGoal-v0",
    entry_point="rlf.envs.ant.ant:AntGoalEnv",
    max_episode_steps=50,
)
