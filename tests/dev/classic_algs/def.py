import sys
sys.path.insert(0, './')
from rlf.algos import QLearning
from rlf.algos import SARSA
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies import DQN
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl.model import MLPBase, reg_mlp_weight_init

class ClassicAlgRunSettings(TestRunSettings):
    def get_policy(self):
        # All methods will use a epsilon-greedy deep Q policy.
        return DQN(lambda n_in, n_out: nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_out)))

    def get_algo(self):
        if self.base_args.alg == 'qlearn':
            return QLearning()
        elif self.base_args.alg == 'sarsa':
            return SARSA()
        else:
            raise ValueError("Unrecognized option {self.base_args.alg}")

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--alg', type=str)

if __name__ == "__main__":
    run_policy(ClassicAlgRunSettings())
