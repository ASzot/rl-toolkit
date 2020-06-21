from rlf.storage.transition_storage import TransitionStorage
import torch
from rlf.algos.base_net_algo import BaseNetAlgo


class OffPolicy(BaseNetAlgo):
    def get_storage_buffer(self, policy, envs, args):
        def convert_to_trans_fn(obs, next_obs, reward, done, masks, bad_masks,
                                ac_info, last_seen):
            return {
                'action': ac_info.action,
                'state': obs,
                'mask': last_seen['masks'],
                'rnn_hxs': last_seen['rnn_hxs'],
                'reward': reward,

                'next_state': next_obs,
                'next_mask': masks,
                'next_rnn_hxs': ac_info.rnn_hxs
            }
        return TransitionStorage(args.trans_buffer_size, convert_to_trans_fn)

    def _sample_transitions(self, storage):
        transitions = storage.sample(self.args.batch_size)
        states = torch.cat([x['state'] for x in transitions])
        actions = torch.cat([x['action'] for x in transitions])
        masks = torch.cat([x['mask'] for x in transitions])
        rnn_hxs = torch.cat([x['rnn_hxs'] for x in transitions])
        rewards = torch.cat([x['reward'] for x in transitions])

        next_states = torch.cat([x['next_state'] for x in transitions])
        next_masks = torch.cat([x['next_mask'] for x in transitions])
        next_rnn_hxs = torch.cat([x['next_rnn_hxs'] for x in transitions])
        if self.args.cuda:
            masks = masks.cuda()
            rewards = rewards.cuda()
        cur_add = {
            'rnn_hxs': rnn_hxs,
            'masks': masks
        }
        next_add = {
            'rnn_hxs': next_rnn_hxs,
            'masks': next_masks
        }
        return states, next_states, actions, rewards, cur_add, next_add

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
        parser.add_argument('--trans-buffer-size', type=int, default=10000)
        parser.add_argument('--batch-size', type=int, default=128)



