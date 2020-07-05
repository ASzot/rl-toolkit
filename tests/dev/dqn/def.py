import sys
sys.path.insert(0, './')
from rlf.algos import QLearning
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies import DQN
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl.model import MLPBasic

class DQNRunSettings(TestRunSettings):
    def get_policy(self):
        return DQN(lambda in_shape: MLPBasic(
            in_shape[0], 64, 2,
            get_activation=lambda: nn.ReLU())
            )

    def get_algo(self):
        return QLearning()

if __name__ == "__main__":
    run_policy(DQNRunSettings())
