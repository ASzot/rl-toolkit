from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.args import str2bool
import numpy as np
import rlf.rl.utils as rutils
import gym


class GymHandWrapper(gym.core.Wrapper):
    def step(self,a):
        obs,reward,done,info = super().step(a)
        info['ep_is_success'] = info['is_success']
        return obs,reward,done,info

class GymHandInterface(EnvInterface):
    def create_from_id(self, env_id):
        env = gym.make(env_id)
        return GymHandWrapper(env)


GYM_HAND_REGISTER_STR = "^(HandReach|HandManipulateBlock|HandManipulateEgg|HandManipulatePen)"
register_env_interface(GYM_HAND_REGISTER_STR, GymHandInterface)
