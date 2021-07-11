"""
Code is heavily based off of https://github.com/denisyarats/pytorch_sac.
The license is at `rlf/algos/off_policy/denis_yarats_LICENSE.md`
"""
import sys
sys.path.insert(0, './')
from rlf.algos.off_policy.sac import SAC
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl.model import BaseNet, IdentityBase, MLPBase
from rlf.policies.actor_critic.dist_actor_q import DistActorQ, get_sac_actor, get_sac_critic
import torch
import math
from functools import partial


class SACRunSettings(TestRunSettings):
    def get_policy(self):
        return DistActorQ(
				get_critic_fn=partial(get_sac_critic, hidden_dim=self.base_args.hidden_dim),
                get_actor_fn=partial(get_sac_actor, hidden_dim=self.base_args.hidden_dim)
                )

    def get_algo(self):
        return SAC()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--hidden-dim', type=int, default=1024)

if __name__ == "__main__":
    run_policy(SACRunSettings())
