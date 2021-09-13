import gym
import gym.spaces as spaces
import rlf.rl.utils as rutils


class SepGoal(gym.Wrapper):
    def __init__(self, env, ndims):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "observation": rutils.reshape_obs_space(
                    self.observation_space, (self.observation_space.shape[0] - ndims,)
                ),
                "desired_goal": rutils.reshape_obs_space(
                    self.observation_space, (ndims,)
                ),
            }
        )
        self.ndims = ndims

    def _trans_obs(self, obs):
        return {
            "observation": obs[: -self.ndims],
            "desired_goal": obs[-self.ndims :],
        }

    def step(self, a):
        obs, reward, done, info = super().step(a)
        obs = self._trans_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        return self._trans_obs(obs)
