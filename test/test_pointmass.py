import random
from argparse import ArgumentParser

import numpy as np
import torch
from rlf.args import data_class_to_args, parse_data_class_from_args
from rlf.envs.pointmass import (LinearPointMassEnv, LinearPointMassParams,
                                PointMassEnv, PointMassParams)


def test_pointmass():
    params = PointMassParams(start_state_noise=0.0, start_idx=0, radius=1)
    batch_size = 2

    env = PointMassEnv(batch_size=batch_size, params=params)
    env.reset()

    for _ in range(1000):
        rnd_ac = torch.tensor(env.action_space.sample())
        rnd_ac = rnd_ac.view(1, -1).repeat(batch_size, 1)
        env.step(rnd_ac)

    batch_size = 1
    params = LinearPointMassParams(start_idx=0)
    env = LinearPointMassEnv(batch_size=batch_size, params=params)
    W = torch.tensor([1.0, 0.0])

    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            ac = obs @ W
            obs, reward, done, info = env.step(ac.view(-1, 1))
            done = done[0]

        assert info[random.randint(0, batch_size - 1)]["ep_dist_to_goal"] < 1e-4


if __name__ == "__main__":
    test_pointmass()
    parser = ArgumentParser()
    data_class_to_args("pm", parser, PointMassParams)
    args = parser.parse_args()

    params = parse_data_class_from_args("pm", args, PointMassParams)
