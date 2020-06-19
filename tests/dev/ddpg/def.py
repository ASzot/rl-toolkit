import sys
sys.path.insert(0, './')
from rlf.algos import DDPG
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import RegA

class DDPGRunSettings(TestRunSettings):
    def get_policy(self):
        return RegActorCritic()

    def get_algo(self):
        return DDPG()

if __name__ == "__main__":
    run_policy(DDPGRunSettings())
