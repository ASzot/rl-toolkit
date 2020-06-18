from rlf.policies.base_net_policy import BaseNetPolicy
import torch.nn as nn
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.rl.model import def_mlp_weight_init
from rlf.rl.model import weight_init


class ActorCritic(BaseNetPolicy):
    """
    Defines an actor and critic type policy
    """

    def __init__(self,
                 get_critic_head_fn=None,
                 get_base_net_fn=None):
        """
        - get_critic_head_fn: (obs_shape: (int), input_shape: (int),
          action_space: gym.spaces.space -> rlf.rl.model.BaseNet)
        """

        # We don't want any base encoder for non-image envs so we can separate
        # critic and actor.
        if get_base_net_fn is None:
            get_base_net_fn = putils.get_img_encoder

        super().__init__(get_base_net_fn)

        if get_critic_head_fn is None:
            get_critic_head_fn = putils.get_def_critic
        self.get_critic_head_fn = get_critic_head_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        obs_shape = rutils.get_obs_shape(obs_space)

        self.critic = self.get_critic_head_fn(obs_shape,
                                              self.base_net.output_shape, action_space)
        critic_head = nn.Linear(self.critic.output_shape[0], 1)

        if putils.is_image_obs(obs_shape):
            def init_(m): return weight_init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
            self.critic_head = init_(critic_head)
        else:
            self.critic_head = def_mlp_weight_init(critic_head)

    def _get_value_from_features(self, base_features, rnn_hxs, masks):
        critic_features, rnn_hxs = self.critic(base_features, rnn_hxs, masks)
        value = self.critic_head(critic_features)
        return value

    def get_value(self, inputs, rnn_hxs, masks):
        base_features, rnn_hxs = self.base_net(inputs, rnn_hxs, masks)
        value = self._get_value_from_features(base_features, rnn_hxs, masks)

        return value

    def get_critic_params(self):
        return list(self.base_net.parameters()) + \
                list(self.critic.parameters()) + \
                list(self.critic_head.parameters())

    def get_actor_params(self):
        return list(self.base_net.parameters()) + \
                list(self.actor_net.parameters()) + \
                list(self.actor_output.parameters())
