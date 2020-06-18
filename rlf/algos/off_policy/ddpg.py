from rlf.algos.off_policy.off_policy_base import OffPolicy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import rlf.algos.utils as autils
from collections import defaultdict


class DDPG(OffPolicy):
    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def _get_optimizers(self):
        return {
                'actor_opt': (
                    optim.Adam(self.policy.get_actor_params(), lr=self.args.lr,
                        eps=self.args.eps),
                    self.policy.get_actor_params,
                    self.args.lr
                ),
                'critic_opt': (
                    optim.Adam(self.policy.get_critic_params(), lr=self.args.critic_lr,
                        eps=self.args.eps),
                    self.policy.get_critic_params,
                    self.args.critic_lr
                )
            }

    def update(self, storage):
        super().update(storage)
        if len(storage) < self.args.warmup_steps:
            return {}

        if len(storage) < self.args.batch_size:
            return {}

        ns = self.get_completed_update_steps(self.update_i)
        if ns < self.args.update_every or ns % self.args.update_every != 0:
            return {}

        avg_log_vals = defaultdict(list)
        for i in range(self.args.update_every):
            log_vals = self._optimize(*self._sample_transitions(storage))
            for k,v in log_vals.items():
                avg_log_vals[k].append(v)

        avg_log_vals = {k: np.mean(v) for k,v in avg_log_vals.items()}

        return avg_log_vals


    def _optimize(self, state, n_state, action, reward, add_info, n_add_info):
        n_masks = n_add_info['masks']
        n_masks = n_masks.to(self.args.device)

        # Get the Q-target
        n_action = self.target_policy(n_state, **n_add_info)
        next_q = self.target_policy.get_value(n_state, n_action, **n_add_info)
        next_q *= n_masks
        next_q = next_q.detach()
        target = reward + (self.args.gamma * next_q)

        # Compute the critic loss. (Just a TD loss)
        q = self.policy.get_value(state, action, **add_info)
        critic_loss = F.mse_loss(q.view(-1), target.view(-1))
        self._standard_step(critic_loss, 'critic_opt')

        # Compute the actor loss
        choose_action = self.policy(state, **add_info)
        actor_loss = -self.policy.get_value(state, choose_action, **add_info).mean()
        self._standard_step(actor_loss, 'actor_opt')

        autils.soft_update(self.policy, self.target_policy, self.args.tau)

        return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)

        #########################################
        # Overrides
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--trans-buffer-size', type=int, default=int(1e6))
        parser.add_argument('--batch-size', type=int, default=int(64))

        #########################################
        # New args
        parser.add_argument('--tau',
            type=float,
            default=1e-3,
            help='Mixture for the target network weight update')

        parser.add_argument('--critic-lr',
            type=float,
            default=1e-3,
            help='Mixture for the target network weight update')

        parser.add_argument('--warmup-steps',
            type=int,
            default=int(1000),
            help='Number of steps before any updates are applied')

        parser.add_argument('--update-every',
            type=int,
            default=int(50),
            help='How many environment steps to do every update')
