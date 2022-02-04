from dataclasses import dataclass

import gym
import numpy as np
import torch
from gym import spaces
from rlf import EnvInterface, register_env_interface
from rlf.args import data_class_to_args, parse_data_class_from_args
from rlf.baselines.vec_env.vec_env import VecEnv
from torch.distributions import Uniform

VEL_LIMIT = 10.0
POS_LIMIT = 1.5
ERROR_MSG = (
    "Make sure rlf.envs.pointmass is imported. Also make sure multiprocessing is forced"
)


@dataclass(frozen=True)
class PointMassParams:
    force_eval_start_dist: bool = False
    force_train_start_dist: bool = True
    clip_bounds: bool = True
    ep_horizon: int = 5
    num_train_regions: int = 4
    start_state_noise: float = np.pi / 20
    dt: float = 0.2
    reward_dist_pen: float = 1 / 10.0
    start_idx: int = -1
    radius: float = np.sqrt(2)


class SingleSampler:
    def __init__(self, point):
        self.point = point

    def sample(self, shape):
        return self.point.unsqueeze(0).repeat(shape[0], 1)


class PointMassEnv(VecEnv):
    def __init__(
        self,
        batch_size,
        params,
        device=None,
        set_eval=False,
        obs_space=None,
        ac_space=None,
    ):
        if device is None:
            device = torch.device("cpu")
        self._batch_size = batch_size
        self._params = params

        self._device = device
        self._goal = torch.tensor([0.0, 0.0]).to(self._device)
        self._ep_step = 0

        self._ep_rewards = []
        if obs_space is None:
            obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if ac_space is None:
            ac_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self._is_eval = set_eval or self._params.force_eval_start_dist
        if self._params.force_train_start_dist:
            self._is_eval = False

        if self._is_eval:
            regions = PointMassEnv.get_regions(0.0, self._params.start_state_noise)
        else:
            regions = PointMassEnv.get_regions(
                np.pi / 4, self._params.start_state_noise
            )

        if self._params.start_state_noise != 0:
            self._start_distributions = Uniform(regions[:, 0], regions[:, 1])
        else:
            self._start_distributions = SingleSampler(regions[:, 0])

        super().__init__(
            self._batch_size,
            obs_space,
            ac_space,
        )

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def forward(self, cur_pos, action):
        action = torch.clamp(action, -1.0, 1.0)
        new_pos = cur_pos + (action * self._params.dt)

        if self._params.clip_bounds:
            new_pos = torch.clamp(new_pos, -POS_LIMIT, POS_LIMIT)
        return new_pos

    def step(self, action):
        self.cur_pos = self.forward(self.cur_pos, action)
        self._ep_step += 1

        is_done = self._ep_step >= self._params.ep_horizon
        dist_to_goal = torch.linalg.norm(
            self._goal - self.cur_pos, dim=-1, keepdims=True
        )

        reward = -self._params.reward_dist_pen * dist_to_goal
        self._ep_rewards.append(reward)

        all_is_done = [is_done for _ in range(self._batch_size)]

        all_info = [
            {"ep_dist_to_goal": dist_to_goal[i].item()} for i in range(self._batch_size)
        ]

        if is_done:
            final_obs = self._get_obs()
            for i in range(self._batch_size):
                all_info[i]["episode"] = {
                    "r": torch.stack(self._ep_rewards).sum(0)[i].item()
                }
                all_info[i]["final_obs"] = final_obs[i]
            self.reset()

        return (self._get_obs(), reward, all_is_done, all_info)

    @staticmethod
    def get_regions(offset, spread):
        inc = np.pi / 2

        centers = [offset + i * inc for i in range(4)]

        return torch.tensor([[center - spread, center + spread] for center in centers])

    def _get_dist_idx(self, batch_size):
        if self._is_eval:
            return torch.randint(0, 4, (batch_size,))
        else:
            if self._params.start_idx == -1:
                return torch.randint(0, self._params.num_train_regions, (batch_size,))
            else:
                return torch.tensor([self._params.start_idx]).repeat(batch_size)

    def _sample_start(self, batch_size, offset_start):
        idx = self._get_dist_idx(batch_size)
        samples = self._start_distributions.sample(idx.shape)
        ang = samples.gather(1, idx.view(-1, 1)).view(-1)

        return (
            torch.stack(
                [
                    self._params.radius * torch.cos(ang),
                    self._params.radius * torch.sin(ang),
                ],
                dim=-1,
            )
            + offset_start
        )

    def reset(self):
        self.cur_pos = self._sample_start(self._batch_size, self._goal)
        self._ep_step = 0
        self._ep_rewards = []

        return self._get_obs()

    def _get_obs(self):
        return self.cur_pos.clone()


@dataclass(frozen=True)
class LinearPointMassParams(PointMassParams):
    dt: float = 0.2
    radius: float = 1.0


class LinearPointMassEnv(PointMassEnv):
    def __init__(
        self,
        batch_size,
        params: LinearPointMassParams = None,
        device=None,
        set_eval=False,
        obs_space=None,
    ):
        if params is None:
            params = LinearPointMassParams()
        if not isinstance(params, LinearPointMassParams):
            raise ValueError("Not correct params")
        super().__init__(
            batch_size,
            params,
            device,
            set_eval,
            obs_space,
            ac_space=spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        )

    def forward(self, cur_pos, action):
        # action = torch.clamp(action, -1.0, 1.0)
        # Change [-1,1] to [-np.pi, 0]
        # action = -np.pi * ((action + 1.0) / 2.0)
        desired_dir = (action + np.pi) % (2 * np.pi)

        delta_pos = torch.cat([torch.cos(desired_dir), torch.sin(desired_dir)], dim=-1)
        new_pos = cur_pos + (delta_pos * self._params.dt)

        if self._params.clip_bounds:
            new_pos = torch.clamp(new_pos, -POS_LIMIT, POS_LIMIT)
        return new_pos

    def _get_obs(self):
        theta = torch.atan2(self.cur_pos[:, 1], self.cur_pos[:, 0])
        r = torch.linalg.norm(self.cur_pos, dim=-1)
        return torch.stack([theta, r], dim=-1)


class PointMassInterface(EnvInterface):
    def requires_tensor_wrap(self) -> bool:
        return False

    def get_setup_multiproc_fn(
        self,
        make_env,
        env_id,
        seed,
        allow_early_resets,
        env_interface,
        set_eval,
        alg_env_settings,
        args,
    ):
        params = parse_data_class_from_args("pm", args, PointMassParams)
        return PointMassEnv(args.num_processes, params, args.device, set_eval)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        data_class_to_args("pm", parser, PointMassParams)


register_env_interface("^RltPointMass", PointMassInterface)
