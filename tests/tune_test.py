#import sys
#sys.path.insert(0, './')
#from rlf.algos.on_policy.ppo import PPO
#from rlf import run_policy
#from tests.test_run_settings import TestRunSettings
#from rlf.policies.actor_critic.dist_actor_critic import DistActorCritic
#import rlf.rl.utils as rutils
#from rlf.ray_test import TrainableTest
#from ray import tune
#import ray
#
#class PPORunSettings(TrainableTest):
#    def get_policy(self):
#        return DistActorCritic()
#
#    def get_algo(self):
#        return PPO()
#
#    def get_config_file(self):
#        return './tests/config.yaml'
#
#ray.init(num_cpus=6)
#tune.run(PPORunSettings,
#        config={
#            'lr': 0.5
#            })
#
