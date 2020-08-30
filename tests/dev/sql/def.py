import sys
sys.path.insert(0, './')
from rlf import run_policy
from rlf.algos.on_policy import SoftQLearning
from rlf.policies.svgd_policy import SVGDPolicy
from tests.test_run_settings import TestRunSettings
from rlf.args import str2bool
from rlf.rl.loggers.plt_logger import PltLogger

class SoftQLearningRunSettings(TestRunSettings):
    def get_policy(self):
        return SoftQLearning()

    def get_algo(self):
        return SVGDPolicy()

run_policy(SoftQLearningRunSettings())
