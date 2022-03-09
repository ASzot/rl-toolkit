import os.path as osp

import pytest
import rlf.envs.pointmass_multigoal
from rlf import run_policy
from rlf.algos import PPO, SAC
from rlf.policies import DistActorCritic, DistActorQ
from rlf.run_settings import RunSettings

NUM_ENV_SAMPLES = 1000
NUM_STEPS = 100
NUM_PROCS = 2


class PPORunSettings(RunSettings):
    def get_config_file(self):
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, "config.yaml")

    def get_policy(self):
        return DistActorCritic()

    def get_algo(self):
        return PPO()


def test_cont_train():
    TEST_ENV = "Pendulum-v1"
    run_settings = PPORunSettings(
        f"--prefix 'ppo-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps {NUM_ENV_SAMPLES} --num-mini-batch 32 --num-epochs 10 --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes {NUM_PROCS} --cuda False"
    )
    run_policy(run_settings)


def test_disc_train():
    TEST_ENV = "Acrobot-v1"
    run_settings = PPORunSettings(
        f"--prefix 'ppo-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps {NUM_ENV_SAMPLES} --num-mini-batch 32 --num-epochs 10 --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes {NUM_PROCS} --cuda False"
    )
    run_policy(run_settings)


def test_full_train():
    TEST_ENV = "MultiGoalRltPointMass-v0"
    run_settings = PPORunSettings(
        f"--prefix 'ppo-test' --lr 0.001 --entropy-coef 0 --num-env-steps 5e5 --num-steps 50 --env-name {TEST_ENV} --log-smooth-len 10 --save-interval -1 --num-processes 32 --cuda False --pm-ep-horizon 50 --pm-dt 0.1 --pm-start-idx 2 --pm-force-train-start-dist True --policy-hidden-dim 64 --normalize-env False --max-grad-norm -1 --force-multi-proc True --pm-start-state-noise 0.05 --eval-interval 1000000000000 --num-render 0 --num-eval 100 --log-interval 25"
    )
    result = run_policy(run_settings)
    assert result.eval_result["ep_success"] > 0.99
