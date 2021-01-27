from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.args import str2bool
import numpy as np
import rlf.rl.utils as rutils
import gym


class GymHandWrapper(gym.core.Wrapper):
    def __init__(self, env, inc_goal):
        super().__init__(env)
        self.inc_goal = inc_goal
        if self.inc_goal:
            o_shape = env.observation_space
            new_shape = o_shape['observation'].shape[0] + o_shape['desired_goal'].shape[0]
            new_shape = (new_shape,)

            self.observation_space.spaces['observation'] = rutils.reshape_obs_space(
                    o_shape['observation'],
                    new_shape)

    def _trans_obs(self, obs):
        if self.inc_goal:
            obs['observation'] = np.concatenate([
                obs['observation'], obs['desired_goal']
                ])
        return obs

    def reset(self):
        obs = super().reset()
        obs = self._trans_obs(obs)
        return obs


    def step(self,a):
        obs,reward,done,info = super().step(a)
        obs = self._trans_obs(obs)
        info['ep_found_goal'] = info['is_success']
        return obs,reward,done,info

class GymHandInterface(EnvInterface):
    def create_from_id(self, env_id):
        if self.args.hand_dense:
            reward_type = 'dense'
        else:
            reward_type = 'sparse'

        env = gym.make(env_id, reward_type=reward_type)
        return GymHandWrapper(env, self.args.hand_inc_goal)

    def get_add_args(self, parser):
        parser.add_argument('--hand-dense', action='store_true',
                default=False)
        parser.add_argument('--hand-inc-goal', action='store_true',
                default=False)


GYM_HAND_REGISTER_STR = "^(HandReach|HandManipulateBlock|HandManipulateEgg|HandManipulatePen)"
register_env_interface(GYM_HAND_REGISTER_STR, GymHandInterface)
