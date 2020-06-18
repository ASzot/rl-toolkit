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

    def update(self, storage):
        if len(storage) < self.args.batch_size:
            return {}

        state, n_state, action, reward, mask, rnn_hxs, add_info, n_add_info = self._sample_transitions(storage)

        next_q_vals = self.policy(n_state).max(1)[0].detach().unsqueeze(-1) * n_add_info['masks']
        target = reward + (next_q_vals * self.args.gamma)
        loss = autils.td_loss(target, self.policy, state, action, add_info)

        self._standard_step(loss)

        return {
                'loss': loss.item()
                }


