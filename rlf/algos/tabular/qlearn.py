from rlf.algos.tabular.base_tabular import BaseTabular
import numpy as np

class TabularQLearning(BaseTabular):
    def update(self, rollout):
        s, n_s, a, r, mask = rollout.get_scalars()
        target = r + self.args.gamma * np.max(self.policy.Q[n_s])
        target *= mask
        tderr = target - self.policy.Q[s,a]
        self.policy.Q[s, a] = self.policy.Q[s,a] + self.args.lr * tderr
        return {
                'td_error': tderr
                }

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--lr', type=float, default=0.1)
