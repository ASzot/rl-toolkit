import gym
import numpy as np
import torch
from gym import spaces
from rlf import EnvInterface, register_env_interface
from rlf.args import str2bool
from rlf.baselines.vec_env.vec_env import VecEnv
from rlf.envs.pointmass import (BatchedTorchPointMassEnvSingleSpawn,
                                PointMassInterface)
from torch.distributions import Uniform

STAGE_1_BONUS = 10.0
STAGE_2_BONUS = 20.0


class MultiGoalBatchedTorchPointMassEnv(BatchedTorchPointMassEnvSingleSpawn):
    """
    Point mass task where the agent should first navigate to 1 goal and then
    navigate back to the starting position.
    """

    def __init__(
        self,
        args,
        set_eval,
    ):
        super().__init__(
            args,
            set_eval,
            obs_space=spaces.Box(low=-2.0, high=2.0, shape=(3,)),
        )

    def _reset_idx(self, idx):
        self.cur_pos[idx] = self._sample_start(1)[0]
        self.cur_vel[idx] = torch.zeros(2)
        self._ep_step[idx] = 0
        self._goal[idx] = torch.tensor([0.0, 0.0])
        self._finished_stage_1[idx] = 0.0
        self._ep_rewards[idx] = []

    def reset(self):
        self._finished_stage_1 = torch.zeros(self._batch_size)
        super().reset()
        self._ep_step = [0 for _ in range(self._batch_size)]
        self._goal = torch.tensor([[0.0, 0.0]]).repeat(self._batch_size, 1)
        self._ep_rewards = {}

        for i in range(self._batch_size):
            self._reset_idx(i)
        return self._get_obs()

    def step(self, action):
        self.cur_pos, self.cur_vel = self.forward(self.cur_pos, self.cur_vel, action)

        dist_to_cur_goal = torch.linalg.norm(
            self._goal - self.cur_pos, dim=-1, keepdims=True
        )

        reward = -(1 / 10.0) * dist_to_cur_goal

        all_is_done = [False for _ in range(self._batch_size)]
        all_info = [{} for i in range(self._batch_size)]

        # Distance between the stage 1 goal and stage 2.
        max_stage_2_dist = torch.linalg.norm(self._goal - self.start_pos, dim=-1)

        at_goal = (
            torch.linalg.norm(self.cur_pos - self._goal, dim=-1)
            < self.args.pm_success_dist
        )
        final_obs = self._get_obs()
        for i in range(self._batch_size):
            self._ep_step[i] += 1
            if self._ep_step[i] >= self.args.pm_ep_horizon:
                all_is_done[i] = True

            all_info[i]["ep_succ_stage2"] = 0.0
            all_info[i]["ep_success"] = 0.0
            if at_goal[i]:
                if self._finished_stage_1[i] == 1.0:
                    # Finished stage 2.
                    reward[i] += STAGE_2_BONUS
                    all_info[i]["ep_succ_stage2"] = 1.0
                    all_info[i]["ep_success"] = 1.0
                    all_is_done[i] = True
                else:
                    # Finished stage 1
                    reward[i] += STAGE_1_BONUS

                self._goal[i] = self.start_pos[i]
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
        return MultiGoalBatchedTorchPointMassEnv(args, set_eval)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(
            "--pm-success-dist",
            type=float,
            default=0.10,
        )
        parser.set_defaults(pm_ep_horizon=50)


register_env_interface("^MultiGoalRltPointMass", MultiGoalPointMassInterface)
