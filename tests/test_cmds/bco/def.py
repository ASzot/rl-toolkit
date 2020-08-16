import sys
sys.path.insert(0, './')
from rlf.algos import BehavioralCloningFromObs
from rlf.policies import BasicPolicy
from rlf import run_policy
from tests.test_run_settings import TestRunSettings
from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
from rlf.rl.model import MLPBase

class BcoRunSettings(TestRunSettings):
    def get_policy(self):
        return BasicPolicy(
                get_base_net_fn=lambda i_shape: MLPBase(
                    i_shape[0], False, (400, 300))
                )

    def get_algo(self):
        return BehavioralCloningFromObs()

if __name__ == "__main__":
    run_policy(BcoRunSettings())
