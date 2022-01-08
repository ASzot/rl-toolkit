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

STAGE_1_BONUS = 5.0
STAGE_2_BONUS = 5.0


class MultiGoalBatchedTorchPointMassEnv(BatchedTorchPointMassEnvSingleSpawn):
    def __init__(
        self,
        fast_env,
        max_num_steps,
        device,
        batch_size,
        pm_start_idx,
        start_noise,
        is_eval,
        num_train_regions,
        should_clip,
        success_dist,
    ):
        self._success_dist = success_dist
        super().__init__(
            fast_env,
            max_num_steps,
            device,
            batch_size,
            pm_start_idx,
            start_noise,
            is_eval,
            num_train_regions,
            should_clip,
        )

    def forward(self, cur_pos, cur_vel, action):
        new_vel = cur_vel + action * 0.2
        new_vel = torch.clip(new_vel, -10.0, 10.0)
        new_pos = cur_pos + (new_vel * self.dt)

        if self._should_clip:
            new_pos = torch.clamp(new_pos, -1.5, 1.5)
        return new_pos, new_vel

    def reset(self):
        self._goal = torch.tensor([[0.0, 0.0]]).repeat(self._batch_size, 1)
        self._finished_stage_1 = torch.tensor([False for _ in range(self._batch_size)])
        return super().reset()

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        obs, reward, done, info = super().step(action)

        # Distance between the stage 1 goal and stage 2.
        max_stage_2_dist = torch.linalg.norm(self._goal - self.start_pos, dim=-1)

        at_goal = (
            torch.linalg.norm(self.cur_pos - self._goal, dim=-1) < self._success_dist
        )
        for i in range(self._batch_size):
            info[i]["ep_succ_stage2"] = 0.0
            if at_goal[i]:
                if self._finished_stage_1[i]:
                    # Finished stage 2.
                    reward[i] += STAGE_2_BONUS
                    done[i] = True
                    info[i]["ep_succ_stage2"] = 1.0
                else:
                    # Finished stage 1
                    reward[i] += STAGE_1_BONUS

                self._goal[i] = self.start_pos[i]
                self._finished_stage_1[i] = True

            stage_1_done = float(self._finished_stage_1[i])
            info[i]["ep_succ_stage1"] = stage_1_done
            # The final distance to the goal is the distance from the agent to
            # the stage 2 goal IF stage 1 is completed. If it is not then it is the
            # distance between the stage 1 goal and the stage 2 goal.
            info[i]["ep_dist_to_stage1"] = info[i]["ep_dist_to_goal"]
            info[i]["ep_dist_to_stage2"] = (
                stage_1_done * info[i]["ep_dist_to_goal"]
                + (1.0 - stage_1_done) * max_stage_2_dist[i].item()
            )

        return obs, reward, done, info


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
        return MultiGoalBatchedTorchPointMassEnv(
            args.pm_fast_env,
            args.pm_ep_horizon,
            args.device,
            args.num_processes,
            args.pm_start_idx,
            args.pm_start_state_noise,
            set_eval or args.pm_force_eval_start_dist,
            args.pm_num_train_regions,
            args.pm_clip,
            args.pm_success_dist,
        )

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(
            "--pm-success-dist",
            type=float,
            default=0.10,
        )


register_env_interface("^MultiGoalRltPointMass", MultiGoalPointMassInterface)
