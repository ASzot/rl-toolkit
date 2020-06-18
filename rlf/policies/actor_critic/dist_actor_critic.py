import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from rlf.policies.base_policy import ActionData
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.policies.actor_critic.base_actor_critic import ActorCritic


class DistActorCritic(ActorCritic):
    """
    Defines an actor/critic where the actor outputs an action distribution
    """

    def __init__(self,
                 get_actor_head_fn=None,
                 get_dist_fn=None,
                 get_critic_head_fn=None,
                 get_base_net_fn=None):
        super().__init__(get_critic_head_fn, get_base_net_fn)
        """
        - get_actor_head_fn: (obs_space : (int), input_shape : (int) ->
          rlf.rl.model.BaseNet)
        """

        if get_actor_head_fn is None:
            get_actor_head_fn = putils.get_def_actor
        self.get_actor_head_fn = get_actor_head_fn

        if get_dist_fn is None:
            get_dist_fn = putils.get_def_dist
        self.get_dist_fn = get_dist_fn

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)
        self.actor = self.get_actor_head_fn(
            rutils.get_obs_shape(obs_space), self.base_net.output_shape)
        self.dist = self.get_dist_fn(
            self.actor.output_shape, self.action_space)

    def get_action(self, state, add_state, rnn_hxs, masks, step_info):
        dist, value, rnn_hxs = self.forward(state, rnn_hxs, masks)
        if self.args.deterministic_policy:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return ActionData(value, action, action_log_probs, rnn_hxs, {
            'dist_entropy': dist_entropy
        })

    def forward(self, inputs, rnn_hxs, masks):
        base_features, rnn_hxs = self.base_net(inputs, rnn_hxs, masks)

        value = self._get_value_from_features(base_features, rnn_hxs, masks)

        actor_features, _ = self.actor(base_features, rnn_hxs, masks)
        dist = self.dist(actor_features)

        return dist, value, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        dist, value, rnn_hxs = self.forward(inputs, rnn_hxs, masks)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()
        return {
            'value': value,
            'log_prob': action_log_probs,
            'ent': dist_entropy,
        }

    def watch(self, logger):
        # Watching currently not supported by PPO
        pass
