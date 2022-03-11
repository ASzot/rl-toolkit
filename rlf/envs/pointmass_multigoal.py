from dataclasses import dataclass

import gym
import numpy as np
import torch
from gym import spaces
from rlf import EnvInterface, register_env_interface
from rlf.args import data_class_to_args, parse_data_class_from_args, str2bool
from rlf.baselines.vec_env.vec_env import VecEnv
from rlf.envs.pointmass import (PointMassEnv, PointMassInterface,
                                PointMassParams)
from torch.distributions import Uniform


@dataclass(frozen=True)
class PointMassMultiGoalParams(PointMassParams):
    success_dist: float = 0.10
    stage1_bonus: float = 10.0
    stage2_bonus: float = 20.0
    early_termination: bool = True

    ep_horizon: int = 50
    dt: float = 0.01


class PointMassMultiGoalEnv(PointMassEnv):
    """
    Point mass task where the agent should first navigate to 1 goal and then
    navigate back to the starting position.
    """

    def __init__(
        self,
        batch_size,
        params,
        device=None,
        set_eval=False,
        obs_space=None,
        ac_space=None,
    ):
        super().__init__(
            batch_size,
            params,
            device,
            set_eval,
            obs_space=spaces.Box(low=-2.0, high=2.0, shape=(3,)),
            ac_space=ac_space,
        )

    def _reset_idx(self, idx):
        self.cur_pos[idx] = self._sample_start(1, torch.zeros(2, device=self._device))[
            0
        ]
        self._ep_step[idx] = 0
        self._goal[idx] = torch.tensor([0.0, 0.0])
        self._finished_stage_1[idx] = 0.0
        self._ep_rewards[idx] = []

    def reset(self):
        """
        Only called once in multiprocessing code.
        """
        self._finished_stage_1 = torch.zeros(self._batch_size).to(self._device)
        super().reset()
        self._ep_step = [0 for _ in range(self._batch_size)]
        self._goal = (
            torch.tensor([[0.0, 0.0]]).repeat(self._batch_size, 1).to(self._device)
        )
        self._stage2_goal = (
            torch.tensor([[-1.0, -1.0]]).repeat(self._batch_size, 1).to(self._device)
        )
        self._ep_rewards = {}

        for i in range(self._batch_size):
            self._reset_idx(i)
        return self._get_obs()

    def step(self, action):
        self.cur_pos = self.forward(self.cur_pos, action)

        dist_to_cur_goal = torch.linalg.norm(
            self._goal - self.cur_pos, dim=-1, keepdims=True
        )

        reward = -(1 / 10.0) * dist_to_cur_goal

        all_is_done = [False for _ in range(self._batch_size)]
        all_info = [{} for i in range(self._batch_size)]

        # Distance between the stage 1 goal and stage 2.
        max_stage_2_dist = torch.linalg.norm(self._goal - self._stage2_goal, dim=-1)

        at_goal = (
            torch.linalg.norm(self.cur_pos - self._goal, dim=-1)
            < self._params.success_dist
        )
        final_obs = self._get_obs()
        for i in range(self._batch_size):
            self._ep_step[i] += 1
            if self._ep_step[i] >= self._params.ep_horizon:
                all_is_done[i] = True

            all_info[i]["ep_succ_stage2"] = 0.0
            all_info[i]["ep_success"] = 0.0
            if at_goal[i]:
                if self._finished_stage_1[i] == 1.0:
                    # Finished stage 2.
                    reward[i] += self._params.stage2_bonus
                    all_info[i]["ep_succ_stage2"] = 1.0
                    all_info[i]["ep_success"] = 1.0
                    if self._params.early_termination:
                        all_is_done[i] = True
                else:
                    # Finished stage 1
                    reward[i] += self._params.stage1_bonus

                self._goal[i] = self._stage2_goal[i]
                self._finished_stage_1[i] = 1.0

            stage_1_done = self._finished_stage_1[i].item()
            all_info[i]["ep_succ_stage1"] = stage_1_done

            # The final distance to the goal is the distance from the agent to
            # the stage 2 goal IF stage 1 is completed. If it is not then it is the
            # distance between the stage 1 goal and the stage 2 goal.
            all_info[i]["ep_dist_to_cur_goal"] = dist_to_cur_goal[i].item()
            all_info[i]["ep_dist_to_stage2"] = (
                stage_1_done * all_info[i]["ep_dist_to_cur_goal"]
                + (1.0 - stage_1_done) * max_stage_2_dist[i].item()
            )

            if all_is_done[i]:
                all_info[i]["episode"] = {
                    "r": sum(self._ep_rewards[i]),
                    "l": self._ep_step[i],
                }
                all_info[i]["final_obs"] = final_obs[i]
                self._reset_idx(i)

            self._ep_rewards[i].append(reward[i].item())

        return (self._get_obs(), reward, all_is_done, all_info)

    def _get_obs(self):
        return torch.cat([self.cur_pos, self._finished_stage_1.view(-1, 1)], dim=-1)


class MultiGoalPointMassInterface(PointMassInterface):
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
        params = parse_data_class_from_args("pm", args, PointMassMultiGoalParams)
        return PointMassMultiGoalEnv(args.num_processes, params, args.device, set_eval)

    def get_add_args(self, parser):
        data_class_to_args("pm", parser, PointMassMultiGoalParams)


register_env_interface("^MultiGoalRltPointMass", MultiGoalPointMassInterface)
