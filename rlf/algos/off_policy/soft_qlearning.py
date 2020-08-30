from rlf.algos.on_policy.off_policy_base import OffPolicy
import rlf.algos.utils as autils


def sample_actions(self, policy, state, add_info, n_particles):
    return torch.stack([policy.forward(state, *add_info)
        for _ in range(n_particles)])

class SoftQLearning(ActorCriticUpdater):
    def init(self, policy, args):
        super().init(policy, args)


    def update(self, rollouts):
        if len(storage) < self.args.batch_size:
            return {}

        state, n_state, action, reward, add_info, n_add_info = self._sample_transitions(storage)

        q_log_vals = self.update_q(state, action, n_state, reward, add_info,
                n_add_info)
        pi_log_vals = self.update_pi(state)

        return {
                **q_log_vals,
                **pi_log_vals
                }


    def update_q(self, state, action, n_state, reward, add_info, n_add_info):
        n_masks = n_add_info['masks']
        cur_q = self.policy.get_value(state, action, **add_info)
        target_actions = sample_actions(self.target_policy, n_state, n_add_info,
                self.args.n_val_particles)


    def update_pi(self, state):
        pass


    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--n-val-particles', type=int, default=16)
        parser.add_argument('--n-kernel-particles', type=int, default=16)
