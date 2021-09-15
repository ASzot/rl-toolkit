import gym
import gym.spaces as spaces
import numpy as np
import rlf.rl.utils as rutils
from rlf.args import str2bool
from rlf.envs.env_interface import EnvInterface, register_env_interface
from rlf.envs.fetch.pick import FetchPickAndPlaceNoise
from rlf.envs.fetch.push import FetchPushNoise


class GoalCheckerWrapper(gym.Wrapper):
    def __init__(self, env, goal_check_cond_fn):
        super().__init__(env)
        self.goal_check_cond_fn = goal_check_cond_fn

    def reset(self):
        self.found_goal = False
        return super().reset()

    def step(self, a):
        obs, reward, done, info = super().step(a)
        self.found_goal = self.found_goal or self.goal_check_cond_fn(self.env, obs)
        if self.found_goal:
            done = True

        info["ep_found_goal"] = float(self.found_goal)
        info["ep_dist_to_goal"] = np.linalg.norm(
            obs["desired_goal"] - obs["achieved_goal"]
        )
        return obs, reward, done, info


class FetchNoVelWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, remove_dict_obs=True):
        super().__init__(env)
        obs_space = self.observation_space.spaces["observation"]
        try:
            self._max_episode_steps = env._max_episode_steps
        except AttributeError:
            pass
        self.remove_dict_obs = remove_dict_obs

        new_obs_space = spaces.Box(
            high=obs_space.high[:-12],
            low=obs_space.low[:-12],
            dtype=obs_space.dtype,
        )
        if self.remove_dict_obs:
            self.observation_space = new_obs_space
        else:
            self.observation_space.spaces["observation"] = new_obs_space

    def observation(self, obs):
        obs["observation"] = obs["observation"][:-15]
        obs["observation"] = np.concatenate([obs["observation"], obs["desired_goal"]])
        if self.remove_dict_obs:
            return obs["observation"]
        else:
            return obs


class BlockGripperActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self._is_success = env.env._is_success
        self.action_space = spaces.Box(
            high=self.action_space.high[:-1],
            low=self.action_space.low[:-1],
            dtype=self.action_space.dtype,
        )

    def step(self, a):
        real_a = np.zeros(len(a) + 1)
        real_a[:-1] = a
        return super().step(real_a)


class GymFetchInterface(EnvInterface):
    def create_from_id(self, env_id):
        if self.args.gf_dense:
            reward_type = "dense"
        else:
            reward_type = "sparse"
        env = gym.make(env_id, reward_type=reward_type)
        return env

    def env_trans_fn(self, env, set_eval):
        env = super().env_trans_fn(env, set_eval)

        try_env = env.env
        if env.env.block_gripper:
            env = BlockGripperActionWrapper(env)

            def check_goal(env, obs):
                return env._is_success(obs["achieved_goal"], obs["desired_goal"])

        else:

            def check_goal(env, obs):
                return env.env._is_success(obs["achieved_goal"], obs["desired_goal"])

        if isinstance(try_env, FetchPickAndPlaceNoise) or isinstance(
            try_env, FetchPushNoise
        ):
            try_env.set_noise_ratio(self.args.noise_ratio, self.args.noise_ratio)

            env = GoalCheckerWrapper(env, check_goal)
        if self.args.fetch_no_vel:
            env = FetchNoVelWrapper(env)
        if self.args.mod_n_steps > 0:
            env.env.env.env._max_episode_steps = self.args.mod_n_steps
            env.env.env._max_episode_steps = self.args.mod_n_steps
        return env

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--gf-dense", type=str2bool, default=True)
        parser.add_argument("--fetch-no-vel", type=str2bool, default=False)
        parser.add_argument("--noise-ratio", type=float, default=1.0)
        parser.add_argument("--mod-n-steps", type=int, default=-1)


FETCH_REGISTER_STR = "^(FetchPickAndPlace|FetchPush|FetchReach|FetchSlide)"
register_env_interface(FETCH_REGISTER_STR, GymFetchInterface)
