import sys
sys.path.insert(0, './')
from rlf import DDPG
from rlf import QLearning
from rlf.algos.off_policy.her import create_her_storage_buff
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf import RegActorCritic
from rlf import DQN
from rlf.rl.model import MLPBase, TwoLayerMlpWithAction
import torch.nn as nn
import torch.nn.functional as F
from rlf.args import str2bool


def reg_init(m):
    return m

def get_actor_head(hidden_dim, action_dim):
    return nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh())

class HerRunSettings(TestRunSettings):
    def get_policy(self):
        hidden_size = 256
        if 'BitFlip' in self.base_args.env_name:
            return DQN(
                    get_base_net_fn=lambda i_shape, recurrent: MLPBase(
                        i_shape[0], False, (hidden_size,),
                        weight_init=reg_init,
                        get_activation=lambda: nn.ReLU()),
                    use_goal=True
                    )
        else:
            return RegActorCritic(
                    get_actor_fn=lambda _, i_shape: MLPBase(
                        i_shape[0], False, (hidden_size, hidden_size),
                        weight_init=reg_init,
                        get_activation=lambda: nn.ReLU()),
                    get_actor_head_fn=get_actor_head,
                    get_critic_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
                        i_shape[0], (hidden_size, hidden_size), a_space.shape[0],
                        weight_init=reg_init,
                        get_activation=lambda: nn.ReLU()),
                    get_critic_head_fn = lambda hidden_dim: nn.Linear(hidden_dim, 1),
                    use_goal=True
                    )

    def get_algo(self):
        pass_kwargs = {}
        if self.base_args.use_her:
            pass_kwargs['create_storage_buff_fn'] = create_her_storage_buff
        if 'BitFlip' in self.base_args.env_name:
            return QLearning(**pass_kwargs)
        else:
            return DDPG(**pass_kwargs)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--use-her', default=True, type=str2bool)

if __name__ == "__main__":
    run_policy(HerRunSettings())
