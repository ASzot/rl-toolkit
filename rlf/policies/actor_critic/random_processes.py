# Code adapted from https://github.com/ShangtongZhang/DeepRL/blob/1dcc02e93c581c0d7004c2e8f22d0f406fac2eb6/deep_rl/component/random_process.py
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np


class NoiseGenerator(object):
    def reset_states(self):
        pass

    def sample(self, cur_episode):
        pass


class GaussianNoise(NoiseGenerator):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self, cur_episode):
        return np.random.randn(*self.size) * self.std(steps=cur_episode)


class UHNoise(NoiseGenerator):
    def __init__(self, size, std, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self, cur_episode):
        use_std = self.std(steps=cur_episode)
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + use_std * np.sqrt(self.dt) * np.random.randn(*self.size)
        )
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


# Taken from https://github.com/ShangtongZhang/DeepRL/blob/1dcc02e93c581c0d7004c2e8f22d0f406fac2eb6/deep_rl/utils/schedule.py
class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None or steps < 0:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class LinearScheduleFixed:
    """
    A version of `LinearSchedule` that decays based solely off of the steps
    parameter. There is no internal state for the current value, unlike
    `LinearSchedule`.
    """

    def __init__(self, start, end, total_steps):
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def __call__(self, steps):
        diff = self.end - self.start
        std = self.start + (diff * (steps / self.total_steps))
        return std
