from rlf.algos.off_policy.off_policy_base import OffPolicy
import rlf.algos.utils as autils
import torch
import torch.nn.functional as F
import torch.nn as nn


class QLearning(OffPolicy):
    """
    Q(s_t, a_t) target:
    r_t + \gamma * max_a Q(s_{t+1}, a)
    """

    def init(self, policy, args):
        super().init(policy, args)
        self.target_policy = self._copy_policy()

    def update(self, storage):
        if len(storage) < self.args.batch_size:
            return {}

        for update_i in range(self.args.updates_per_batch):
            state, n_state, action, reward, add_info, n_add_info = self._sample_transitions(storage)

            next_q_vals = self.policy(n_state, **n_add_info).max(1)[0].detach().unsqueeze(-1) * n_add_info['masks']
            target = reward + (next_q_vals * self.args.gamma)
            loss = autils.td_loss(target, self.policy, state, action, add_info)

            self._standard_step(loss)

        autils.soft_update(self.policy, self.target_policy, self.args.tau)

        return {
                'loss': loss.item()
                }


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--tau', type=float, default=1e-3,
            help=("Mixture for the target network weight update. ",
                "If non-zero this is DDQN"))

        parser.add_argument('--updates-per-batch', type=int, default=1,
            help='Number of updates to perform in each call to update')
