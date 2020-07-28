from rlf.envs.env_interface import EnvInterface, register_env_interface
import gym
import numpy as np

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass

class FlatGrid(gym.Wrapper):
    def _proc_obs(self, x):
        return x['image'].reshape(-1)

    def __init__(self, env):
        super().__init__(env)
        ob_space = self.observation_space['image']
        self.observation_space = gym.spaces.Box(
                shape=(np.prod(ob_space.shape),),
                low=ob_space.low.reshape(-1)[0],
                high=ob_space.high.reshape(-1)[0])

    def reset(self):
        obs = super().reset()
        obs = self._proc_obs(obs)
        return obs

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = self._proc_obs(obs)
        return obs, reward, done, info

class MinigridInterface(EnvInterface):
    def create_from_id(self, env_id):
        env = gym.make(env_id)
        if self.args.gw_flatten:
            env = FlatGrid(env)
        return env

    def get_add_args(self, parser):
        parser.add_argument('--gw-flatten', type=str2bool, default=True)

register_env_interface("^MiniGrid", MinigridInterface)
