import torch
import rlf.policies.utils as putils
from abc import ABC, abstractmethod
from dataclasses import dataclass
import rlf.rl.utils as rutils

def create_simple_action_data(action, extra={}):
    """
    Create some policy output that consists of just the action. This means no
    need to also return teh value, action_log_probs, etc.
    """
    return ActionData(torch.tensor(0.0), action,
            torch.zeros([*action.shape[:-1], 1]),
            torch.tensor([0]), extra, 0)

def create_np_action_data(action):
    return create_simple_action_data(torch.tensor([[action]]))

class ActionData(object):
    """
    Object returned on every get_action. Note that you don't need to fill out
    every field, see `create_simple_action_data` for more.
    """
    def __init__(self, value, action, action_log_probs, rnn_hxs, extra, add_reward=0):
        self.value = value
        self.action = action
        self.action_log_probs = action_log_probs
        self.rnn_hxs = rnn_hxs
        self.add_reward = add_reward
        self.extra = extra
        self.take_action = torch.tensor(action.cpu().numpy())

    def clip_action(self, low_bound, upp_bound):
        # When CUDA is enabled the action will be on the GPU.
        self.action = rutils.multi_dim_clip(self.action,
                low_bound.to(self.action.device),
                upp_bound.to(self.action.device))
        self.take_action = rutils.multi_dim_clip(self.take_action, low_bound, upp_bound)

@dataclass
class StepInfo:
    cur_num_steps: int
    cur_num_episodes: int
    is_eval: bool


def get_step_info(n_full_iter, within_loop_iter, episode_count, args):
    return StepInfo(
            (n_full_iter * args.num_steps + within_loop_iter) * args.num_processes,
            episode_count,
            False)

def get_empty_step_info():
    return StepInfo(None, None, True)


class BasePolicy(ABC):
    """
    Foundation for all RL policies to derive from. Defines basic behavior which
    could be needed. Agents do not need to necessarily implement all method.
    """

    def __init__(self):
        pass

    def init(self, obs_space, action_space, args):
        self.action_space = action_space
        self.obs_space = obs_space
        self.args = args

    def get_add_args(self, parser):
        parser.add_argument('--deterministic-policy', action='store_true',
                default=False)

    def load_from_checkpoint(self, checkpointer):
        pass

    def save_to_checkpoint(self, checkpointer):
        pass

    def load_resume(self, checkpointer):
        pass

    @abstractmethod
    def get_action(self, state, add_state, rnn_hxs, masks, step_info):
        """
        - step_info: Dictionary consisting of keys
          {
              'cur_num_steps',
              'cur_num_episodes'
          }
        - add_state: If the state is a dictionary, this contains all the other
          non 'observation' keys.
        Return: ActionData
        """
        pass

    def watch(self, logger):
        """
        Set up underlying parameters to be monitored some logger service.
        """
        pass

    def to(self, device):
        # Dummy return
        return self

    def eval(self):
        pass

    def train(self):
        pass
