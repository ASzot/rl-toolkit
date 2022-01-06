import operator
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import *
from typing import Any, Dict, Tuple

import numpy as np
import rlf.rl.utils as rutils
import torch
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage import BaseStorage, RolloutStorage


class BaseIRLAlgo(BaseILAlgo):
    """
    For algorithms with learned reward functions. This class provides
    functionality to track inferred rewards in episodes, clear out environment
    rewards, and make updating easier.

    `get_reward` and `_update_reward_func` should be overriden. Example algorithm for learning a linear reward:

    class MyIRLAlgo(BaseIRLAlgo):
        def init(self, policy, args):
            # Put any setup here.
            super().init(policy, args)
            ob_dim = self.policy.obs_space.shape[0]
            ac_dim = self.action_space.shape[0]

            self.learned_reward_net = nn.Linear(ob_dim+ac_dim, 1)

        def get_reward(self, state, next_state, action, mask, add_inputs):
            # torch.no_grad wraps this, so no need to call torch.no_grad from
            # within
            input = torch.cat([state, action], dim=-1)
            return self.learned_reward_net(input), {
                "value_to_log_from_reward_inference": 0.0
                }

        def _update_reward_func(self, storage: BaseStorage) -> Dict[str, Any]:
            batch_sampler = storage.get_generator(
                # traj_batch_size is the default batch size for IL updates.
                mini_batch_size=self.args.traj_batch_size,
            )

            for batch in batch_sampler:
                input = torch.cat([batch['state'], batch['actions']], dim=-1)
                pred_reward = self.learned_reward_net(input)
                # Update reward somehow...

            return {
                # Whatever values you want to plot.
                "loss": 0.0
                }

        def get_add_args(self, parser):
            super().get_add_args(parser)
            # Add any command line arguments
            parser.add_argument("--custom-command-line-arg", type=float, default=0.1)

    """

    def __init__(self, exp_generator=None):
        super().__init__(exp_generator=exp_generator)
        self.traj_log_stats = defaultdict(list)

    def init(self, policy, args):
        super().init(policy, args)
        self.ep_log_vals = defaultdict(lambda: deque(maxlen=args.log_smooth_len))
        self.culm_log_vals = defaultdict(
            lambda: [0.0 for _ in range(args.num_processes)]
        )

    @abstractmethod
    def get_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        mask: torch.Tensor,
        add_info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Infers the reward for the batch of data. Let `N` be the number of
        rollout steps, `S` be the state dimension, and `A` be the action
        dimension.

        :param state: Shape (N, S)
        :param next_state: Shape (N, S)
        :param action: Shape (N, A)
        :param mask: Shape (N, 1). 0 if the episode ended between state and next state, 1 otherwise.
        :param add_info: Additional information about the current state such as goals.
        :returns: Shape (N, 1) of the reward for each state along with a dictionary of any values to be logged.
        """
        raise NotImplementedError()

    def _update_reward_func(self, storage: BaseStorage) -> Dict[str, Any]:
        """
        :returns: Dictionary of values to be logged.
        """
        return {}

    def _trans_agent_state(self, state, other_state=None):
        return rutils.get_def_obs(state)

    def _infer_rollout_storage_reward(self, storage, log_vals):
        add_info = {k: storage.get_add_info(k) for k in storage.get_extract_info_keys()}
        for k in storage.ob_keys:
            if k is not None:
                add_info[k] = storage.obs[k]

        for step in range(self.args.num_steps):
            mask = storage.masks[step]
            state = self._trans_agent_state(storage.get_obs(step))
            next_state = self._trans_agent_state(storage.get_obs(step + 1))
            action = storage.actions[step]
            add_inputs = {k: v[(step + 1) - 1] for k, v in add_info.items()}

            with torch.no_grad():
                rewards, ep_log_vals = self.get_reward(
                    state, next_state, action, mask, add_inputs
                )

            ep_log_vals["reward"] = rewards
            storage.rewards[step] = rewards

            for i in range(self.args.num_processes):
                for k in ep_log_vals:
                    self.culm_log_vals[k][i] += ep_log_vals[k][i].item()

                if storage.masks[step, i] == 0.0:
                    for k in ep_log_vals:
                        self.ep_log_vals[k].append(self.culm_log_vals[k][i])
                        self.culm_log_vals[k][i] = 0.0

        for k, vals in self.ep_log_vals.items():
            log_vals[f"culm_irl_{k}"] = np.mean(vals)

    def update(self, storage):
        super().update(storage)
        is_rollout_storage = isinstance(storage, RolloutStorage)
        if is_rollout_storage:
            # CLEAR ALL REWARDS so no environment rewards can leak to the IRL method.
            for step in range(self.args.num_steps):
                storage.rewards[step] = 0

        log_vals = self._update_reward_func(storage)

        if is_rollout_storage:
            self._infer_rollout_storage_reward(storage, log_vals)
        else:

            def get_reward(states, actions, next_states, mask):
                with torch.no_grad():
                    return self.get_reward(states, next_states, actions, mask, {})[0]

            storage.set_modify_reward_fn(get_reward)

        return log_vals

    def on_traj_finished(self, trajs):
        pass

    def get_add_args(self, parser):
        super().get_add_args(parser)

        parser.add_argument(
            "--freeze-reward",
            type=str2bool,
            default=False,
            help="""
                If true, the learned reward is not updated.
            """,
        )
