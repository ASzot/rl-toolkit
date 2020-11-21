from rlf.storage.transition_storage import TransitionStorage
import torch
from rlf.algos.base_net_algo import BaseNetAlgo
import rlf.rl.utils as rutils

class OffPolicy(BaseNetAlgo):
    def __init__(self):
        super().__init__()

    def init(self, policy, args):
        args.trans_buffer_size = int(args.trans_buffer_size)
        super().init(policy, args)

    def get_storage_buffer(self, policy, envs, args):
        return TransitionStorage(policy.obs_space, policy.action_space,
                args.trans_buffer_size, args)

    def _sample_transitions(self, storage):
        return storage.sample_tensors(self.args.batch_size)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        parser.add_argument('--num-processes', type=int, default=1)
        parser.add_argument('--num-steps', type=int, default=1)
        # Off-policy algorithms have far more frequent updates.
        ADJUSTED_INTERVAL = 1000
        parser.add_argument('--log-interval', type=int,
                            default=ADJUSTED_INTERVAL)
        parser.add_argument('--save-interval', type=int,
                            default=50*ADJUSTED_INTERVAL)
        parser.add_argument('--eval-interval', type=int,
                            default=50*ADJUSTED_INTERVAL)

        #########################################
        # New args
        parser.add_argument('--trans-buffer-size', type=float, default=10000)
        parser.add_argument('--batch-size', type=int, default=128)

        #########################################
        # HER related. Ideally they would be in the `HerStorage` object. This is
        # a temporally place for them.
        parser.add_argument('--her-K', type=int, default=3)
        parser.add_argument('--her-strat', type=str, default='future',
                help="Valid options are ['future', 'final']")
