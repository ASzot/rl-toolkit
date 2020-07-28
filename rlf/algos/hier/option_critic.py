from rlf.algos.base_net_algo import BaseNetAlgo
from rlf.storage.transition_storage import TransitionStorage
from rlf.storage.rollout_storage import RolloutStorage

class OptionCritic(BaseNetAlgo):
    def get_storage_buffer(self, policy, envs, args):
        return NestedStorage(
                {
                    'replay_buffer': TransitionStorage(buff_size, args),
                    'on_policy': RolloutStorage(args.num_steps,
                        args.num_processes, envs.observation_space,
                        envs.action_space, args)
                    }, 'on_policy')

    def update(self, storage):
        import ipdb; ipdb.set_trace()
        print('here!')

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides

        #########################################
        # New args
        parser.add_argument('--trans-buffer-size', type=int, default=10000)
        parser.add_argument('--batch-size', type=int, default=128)
