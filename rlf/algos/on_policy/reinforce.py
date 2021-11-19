from collections import defaultdict

import numpy as np
import torch.optim as optim
from rlf.algos.on_policy.on_policy_base import OnPolicy


class REINFORCE(OnPolicy):
    def update(self, rollouts):
        self._compute_returns(rollouts)
        log_vals = defaultdict(list)
        advantages = rollouts.compute_advantages().detach()

        for e in range(self._arg("num_epochs")):
            data_generator = rollouts.get_generator(
                advantages, self._arg("num_mini_batch")
            )
            for sample in data_generator:
                ac_eval = self.policy.evaluate_actions(
                    sample["state"],
                    sample["other_state"],
                    sample["hxs"],
                    sample["mask"],
                    sample["action"],
                )

                loss = -ac_eval["log_prob"] * sample["return"]
                loss = loss.mean()

                self._standard_step(loss)

                log_vals["loss"].append(loss.item())
        log_vals = {k: np.mean(v) for k, v in log_vals.items()}
        return log_vals
