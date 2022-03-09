import os.path as osp

import pytest
import rlf.envs.pointmass_multigoal
from rlf import run_policy
from rlf.algos import PPO, SAC
from rlf.policies import DistActorCritic, DistActorQ
from rlf.run_settings import RunSettings

NUM_ENV_SAMPLES = 1000
NUM_STEPS = 1


class SacRunSettings(RunSettings):
    def get_config_file(self):
        config_dir = osp.dirname(osp.realpath(__file__))
        return osp.join(config_dir, "config.yaml")

    def get_policy(self):
        return DistActorQ()

    def get_algo(self):
        return SAC()


def test_sac_cont_train():
    TEST_ENV = "Pendulum-v1"
    run_settings = SacRunSettings(
        f"--prefix 'sac-test' --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --num-env-steps {NUM_ENV_SAMPLES} --num-steps {NUM_STEPS} --env-name {TEST_ENV} --eval-interval -1 --log-smooth-len 10 --save-interval -1 --num-processes 1 --cuda False --n-rnd-steps 10"
    )
    run_policy(run_settings)


def test_full_train():
    run_settings = SacRunSettings(
        f"--env-name MultiGoalRltPointMass-v0 --prefix sac-test --eval-interval 10000000000 --num-eval 100 --prefix sac --policy-hidden-dim 64 --dist-q-hidden-dim 64 --normalize-env False --max-grad-norm -1 --num-env-steps 8e4 --save-interval -1 --cuda False --pm-ep-horizon 50 --log-interval 1000 --pm-start-state-noise 0.05 --pm-dt 0.1 --pm-start-idx 2 --pm-force-train-start-dist True --trans-buffer-size 5e4 --batch-size 256 --force-multi-proc True --alpha-lr 0.001 --critic-lr 0.0003 --lr 0.001 --eval-num-processes 32 --num-render 0"
    )
    result = run_policy(run_settings)
    assert result.eval_result["ep_success"] > 0.99
