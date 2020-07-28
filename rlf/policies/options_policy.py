from rlf.policies.base_net_policy import OptionCriticPolicy
import rlf.policies.utils as putils
from torch.distributions import Categorical, Bernoulli
import torch.nn as nn
from rlf.policies.base_policy import ActionData

class OptionsPolicy(BaseNetPolicy):
    def __init__(self,
            get_critic_fn=None,
            get_term_fn=None,
            get_option_fn=None,
            use_goal=False,
            get_base_net_fn=None):
        # We don't want any base encoder for non-image envs so we can separate
        # the neural networks for the intra-option, option and termination
        # policies. We can use a shared image encoding when working with images
        if get_base_net_fn is None:
            get_base_net_fn = putils.get_img_encoder

        if get_critic_fn is None:
            get_critic_fn = putils.get_mlp_net_fn((64, 64, 1))
        if get_term_fn is None:
            get_term_fn = putils.get_mlp_net_var_out_fn((64, 64))
        if get_option_fn is None:
            get_option_fn = putils.get_mlp_net_var_out_fn((64, 64))

        super().__init__(use_goal, get_base_net_fn)
        self.get_critic_fn = get_critic_fn
        self.get_term_fn = get_term_fn
        self.get_option_fn = get_option_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        in_shape = self.base_net.output_shape

        if not isinstance(action_space, spaces.Discrete):
            raise ValueError(("Currently option critic only supports discrete",
                    "action space environments. The change to make it work with
                    "continuous action space envrionments is actually very",
                    "easy, so it won't be hard to add."))

        self.critic_net = self.get_critic_fn(in_shape)
        self.term_net = self.get_term_fn(in_shape, self.args.n_options)

        self.option_nets = nn.ModuleList([
            self.get_option_fn(in_shape, action_space.n)
            for _ in range(self.args.n_options)
            ])

    def _sel_option(self, state, add_state, rnn_hxs, masks, eps_threshold):
        if np.random.rand() < eps_threshold:
            return torch.LongTensor([np.random.choice(self.args.num_options)
                for i in range(state.shape[0])])
        else:
            return self.critic_net(state, add_state, rnn_hxs,
                    masks).argmax(dim=-1)

    def get_action(self, state, add_state, hxs, masks, step_info):
        num_steps = step_info.cur_num_steps
        eps_threshold = self.args.eps_end + \
                (self.args.eps_start - self.args.eps_end) * \
                math.exp(-1.0 * num_steps / self.args.eps_decay)

        if 'option' in hxs:
            prev_option = hxs['option']
            term = hxs['term']
        elif not (masks == 0).all():
            raise ValueError(("Hidden state not set, not sure which",
                "option to use"))

        # If the mask is 0, the episode is over and we must decide a new
        # option.
        new_option_sel = self._sel_option(state, add_state, rnn_hxs, masks,
                eps_threshold)

        cur_option = torch.zeros(len(self.args.num_options))

        for i, mask in enumerate(masks):
            if mask[i] == 0 or term[i] == 1.0:
                cur_option[i] = new_option_sel[i]
            else:
                cur_option[i] = prev_option[i]

        logits = self.option_nets[cur_option](state, add_state, rnn_hxs, masks)
        action_dist = Categorical(logits)

        action = action_dist.sample()
        action_log_probs = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy()

        term_logit = self.term_net(state, add_state, rnn_hxs, masks)[cur_option]
        term = Bernoulli(term_logit).sample()

        hxs = {
                'option': cur_option,
                'term': term,
                }

        return ActionData(value, action, action_log_probs, hxs, {
            'dist_entropy': dist_entropy
        })

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--n-options", type=int, default=3,
            help="Number of option sub-policies")
        parser.add_argument('--eps-start', type=float, default=0.9)
        parser.add_argument('--eps-end', type=float, default=0.05)
        parser.add_argument('--eps-decay', type=float, default=200)
