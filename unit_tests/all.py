import sys
sys.path.insert(0, '../')
import pytest
from tests.test_cmds.ppo.main import PPORunSettings

@pytest.fixture
def base_args():
    cmds = [
            '--num-processes 1',
            '--eval-num-processes 1',
            '--no-wb',
            '--cuda False',
            ]
    return ' '.join(cmds)

# Test PPO state / images
def test_ppo(base_args):
    result = run_policy(PPORunSettings(base_args + ' '.join([
        ' --env-name CartPole-v0',
        ' --num-steps 4',
        ' --num-epochs 2',
        ])))
    import ipdb; ipdb.set_trace()


# Test GAIL state  / images

# Test DQN state / images

# Test evaluation only

# Test saving a policy and then loading a policy

# Test evaluation on an interval

# Test BC

# Test GAIfO

