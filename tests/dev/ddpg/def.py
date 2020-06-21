import sys
sys.path.insert(0, './')
from rlf.algos import DDPG
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies import RegActorCritic
from rlf.rl.model import MLPBase, TwoLayerMlpWithAction
import torch.nn as nn
import torch.nn.functional as F


def uniform_init(m, init_w=3e-3):
    m.weight.data.uniform_(-init_w, init_w)
    m.bias.data.uniform_(-init_w, init_w)
    return m

def reg_init(m):
    return m

class DDPGRunSettings(TestRunSettings):
    def get_policy(self):
        hidden_size = 256
        return RegActorCritic(
                output_activation = F.tanh,
                get_actor_head_fn=lambda _, i_shape: MLPBase(
                    i_shape[0], False, (hidden_size, hidden_size),
                    weight_init=reg_init,
                    get_activation=lambda: nn.ReLU()),
                get_critic_head_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
                    i_shape[0], (hidden_size, hidden_size), a_space.shape[0],
                    weight_init=reg_init,
                    act_fn=F.relu)
                )

    def get_algo(self):
        return DDPG()

if __name__ == "__main__":
    run_policy(DDPGRunSettings())
