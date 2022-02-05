import os.path as osp

from rlf import RunSettings, run_policy
from rlf.algos import (AIRL, DDPG, GAIFO, GAIL, PPO, SAC, SQIL, BaseAlgo,
                       BehavioralCloning, BehavioralCloningFromObs, QLearning)
from rlf.policies import (BasicPolicy, DistActorCritic, DistActorQ,
                          RandomPolicy, RegActorCritic)

IL_METHODS = {
    ###############
    # IL methods
    "gail_ppo": (GAIL, DistActorCritic),
    "gaifo_ppo": (GAIFO, DistActorCritic),
    "airl_ppo": (AIRL, DistActorCritic),
    "bc": (BehavioralCloning, BasicPolicy),
    "bco": (BehavioralCloningFromObs, BasicPolicy),
    "sqil": (SQIL, DistActorQ),
}

RL_METHODS = {
    ###############
    # RL methods
    "ppo": (PPO, DistActorCritic),
    "sac": (SAC, DistActorQ),
    "ddpg": (DDPG, RegActorCritic),
    "rnd": (BaseAlgo, RandomPolicy),
}

METHODS = {**IL_METHODS, **RL_METHODS}


class ExampleRunSettings(RunSettings):
    def get_algo(self):
        return METHODS[self.base_args.alg][0]()

    def get_policy(self):
        return METHODS[self.base_args.alg][1]()

    def get_add_args(self, parser):
        parser.add_argument("--alg", type=str, required=True)

    def get_config_file(self) -> str:
        return osp.join(osp.dirname(osp.realpath(__file__)), "example_config.yaml")


if __name__ == "__main__":
    run_policy(ExampleRunSettings())
