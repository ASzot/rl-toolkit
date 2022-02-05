import os
import signal

import pytest
from examples.train import IL_METHODS, RL_METHODS, ExampleRunSettings
from rlf import run_policy


@pytest.mark.parametrize("method_name", [x for x in IL_METHODS.keys()])
def test_il_methods(method_name):
    env_name = "Pendulum-v1"
    run_settings = ExampleRunSettings(
        f"--alg {method_name} --env-name {env_name} --cuda False --num-env-steps 5e3 --save-interval -1 --eval-interval -1 --traj-load-path examples/expert_demonstrations/pendulum_test.pt"
    )
    run_policy(run_settings)


@pytest.mark.parametrize("method_name", [x for x in RL_METHODS.keys()])
def test_rl_methods(method_name):
    env_name = "Pendulum-v1"
    run_settings = ExampleRunSettings(
        f"--alg {method_name} --env-name {env_name} --cuda False --num-env-steps 5e3 --save-interval -1 --eval-interval -1"
    )
    run_policy(run_settings)
