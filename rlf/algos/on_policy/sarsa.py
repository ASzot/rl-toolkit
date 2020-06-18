import torch
import torch.nn as nn
from rlf.algos.on_policy.on_policy_base import OnPolicy
import rlf.algos.utils as autils


class SARSA(OnPolicy):
    """
    Q(s_t, a_t) target:
    r_t + \gamma * Q(s_{t+1}, a_{t+1})
    """
    def update(self, rollouts):
        # n_ stands for "next"
        state, action, reward, n_state, n_action, rest = rollouts.get_sarsa_rollout_data()

        next_q_vals = self.policy(n_state).gather(1, n_action)
        next_q_vals *= rest['mask'][1:]

        target = reward + self.args.gamma * next_q_vals
        target = target.detach()

        loss = autils.td_loss(target, self.policy, state, action, (None, None))
        self._standard_step(loss)

        return {
                'loss': loss.item()
                }

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        parser.add_argument('--num-processes', type=int, default=1)
