from rlf.algos.off_policy.off_policy_base import ExperienceSampler
import torch

class HerSampler(ExperienceSampler):
    def convert_to_trans_fn(self, obs, next_obs, reward, done, masks, bad_masks,
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

    def sample_transitions(self, sample_size):
        transitions = storage.sample()
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
