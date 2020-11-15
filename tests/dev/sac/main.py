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
from torch import distributions as pyd
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl.model import BaseNet, IdentityBase, MLPBase
from rlf.policies.actor_critic.dist_actor_q import DistActorQ
import torch
import math
import pybulletgym

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    return m

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class SquashedDiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds

        self.fc_dist = weight_init(nn.Linear(num_inputs, 2*num_outputs))

    def forward(self, x):
        mu, log_std = self.fc_dist(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class DoubleQCritic(BaseNet):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__(False, None, hidden_dim)

        dims = [hidden_dim] * hidden_depth
        dims.append(1)

        self.Q1 = MLPBase(obs_dim + action_dim, False, dims,
                weight_init=weight_init, get_activation=lambda: nn.ReLU(),
                no_last_act=True)
        self.Q2 = MLPBase(obs_dim + action_dim, False, dims,
                weight_init=weight_init, get_activation=lambda: nn.ReLU(),
                no_last_act=True)

    @property
    def output_shape(self):
        return (2,)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1, _ = self.Q1(obs_action, None, None)
        q2, _ = self.Q2(obs_action, None, None)

        return q1, q2


class SACRunSettings(TestRunSettings):
    def get_policy(self):
        def get_sac_dist(in_shape, ac_space, log_std_bounds):
            return SquashedDiagGaussian(in_shape[0], ac_space.shape[0],
                    log_std_bounds)
        def get_sac_critic(obs_shape, in_shape, action_space):
            return DoubleQCritic(in_shape[0], action_space.shape[0], 1024, 2)
        def get_base_net(i_shape):
            return MLPBase(i_shape[0], False, (1024, 1024), get_activation=lambda: nn.ReLU())


        return DistActorQ(
                get_dist_fn=get_sac_dist,
				get_critic_fn=get_sac_critic,
                get_base_net_fn=get_base_net
                )

    def get_algo(self):
        return SAC()

    def get_add_args(self, parser):
        super().get_add_args(parser)

if __name__ == "__main__":
    run_policy(SACRunSettings())
