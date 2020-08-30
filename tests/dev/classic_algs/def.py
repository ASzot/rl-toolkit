import sys
sys.path.insert(0, './')
from rlf.algos.tabular.qlearn import TabularQLearning
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.tabular.q_table import QTable
from rlf.rl.loggers.plt_logger import PltLogger

class ClassicAlgRunSettings(TestRunSettings):
    def get_policy(self):
        return QTable()

    def get_algo(self):
        if self.base_args.alg == 'qlearn':
            return TabularQLearning()
        #elif self.base_args.alg == 'sarsa':
        #    return TabularSARSA()
        else:
            raise ValueError("Unrecognized option {self.base_args.alg}")

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--alg', type=str)

    def get_logger(self):
        return PltLogger(['eval_train_r'], '# Updates', ['Reward'], ['Frozen Lake'])

if __name__ == "__main__":
    run_policy(ClassicAlgRunSettings())
