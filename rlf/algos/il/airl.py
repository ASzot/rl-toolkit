"""
This code is based on these resources:
- https://github.com/ku2482/gail-airl-ppo.pytorch/blob/master/gail_airl_ppo/network/disc.py
- https://github.com/justinjfu/inverse_rl/blob/master/inverse_rl/models/airl_state.py
"""
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.gaifo import GaifoDiscrim
from rlf.algos.il.gail import GailDiscrim
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool


class AIRL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None, exp_generator=None):
        super().__init__(
            [AirlDiscrim(get_discrim, exp_generator=exp_generator), agent_updater], 1
        )


class AirlNetDiscrim(nn.Module):
    def __init__(self, state_enc, args):
        super().__init__()
        self.state_enc = state_enc.net
        hidden_dim = args.gail_disc_hidden_dim

        self.g = nn.Sequential(
            nn.Linear(state_enc.output_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.h = nn.Sequential(
            nn.Linear(state_enc.output_shape[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.args = args

    def reward_forward(
        self, states, actions, mask, next_states, policy, base_reward_only=False
    ):
        log_p = self.f(states, actions, mask, next_states, policy, not base_reward_only)
        if base_reward_only:
            return log_p

        with torch.no_grad():
            log_q = policy.evaluate_actions(states, {}, {}, mask, actions)["log_prob"]

        logits = log_p - (self.args.airl_entropy_bonus * log_q)
        s = torch.sigmoid(logits)
        eps = 1e-20
        return (s + eps).log() - (1 - s + eps).log()

    def f(self, states, actions, mask, next_states, policy, can_use_shaped=True):
        states_enc = self.state_enc(states)
        rs = self.g(states_enc)

        if self.args.airl_reward_shaping and can_use_shaped:
            next_states_enc = self.state_enc(next_states)
            vs = self.h(states_enc)
            next_vs = self.h(next_states_enc)

            return rs + (self.args.gamma * mask.view(-1, 1) * next_vs) - vs
        else:
            return rs

    def discrim_forward(self, states, actions, mask, next_states, policy):
        log_p = self.f(states, actions, mask, next_states, policy)
        with torch.no_grad():
            log_q = policy.evaluate_actions(states, {}, {}, mask, actions)["log_prob"]

        return log_p - log_q


class AirlDiscrim(GailDiscrim):
    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        base_net = self._get_base_net_fn(ob_shape)

        return AirlNetDiscrim(base_net, self.args).to(self.args.device)

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(
            mini_batch_size=self.expert_train_loader.batch_size,
            from_recent=self.args.off_policy_recent,
            num_samples=self.args.off_policy_count,
            get_next_state=True,
        )
        return self.expert_train_loader, agent_experience

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        d = self.args.device
        exp_s0 = self._norm_expert_state(expert_batch["state"], obsfilt).float()
        exp_s1 = self._norm_expert_state(expert_batch["next_state"], obsfilt).float()
        expert_actions = expert_batch["actions"].to(d)
        expert_actions = self._adjust_action(expert_actions)
        expert_mask = (1 - expert_batch["done"]).to(d)

        agent_s0 = agent_batch["state"].to(d)
        agent_s1 = agent_batch["next_state"].to(d)
        agent_actions = agent_batch["action"].to(d)
        agent_mask = agent_batch["mask"].to(d)

        expert_d = self.discrim_net.discrim_forward(
            exp_s0, expert_actions, expert_mask, exp_s1, self.policy
        )
        agent_d = self.discrim_net.discrim_forward(
            agent_s0, agent_actions, agent_mask, agent_s1, self.policy
        )
        return expert_d, agent_d, 0

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        settings.include_info_keys.extend(
            [("final_obs", lambda env: rutils.get_obs_shape(env.observation_space))]
        )
        return settings

    def _compute_discrim_reward(self, state, next_state, action, mask, add_inputs):
        if next_state is None:
            # We must be visualizing a single state reward, so don't use
            # shaping term
            return self.discrim_net.reward_forward(
                state, action, mask, next_state, self.policy, base_reward_only=True
            )

        finished_episodes = [i for i in range(len(mask)) if mask[i] == 0.0]
        obsfilt = self.get_env_ob_filt()
        for i in finished_episodes:
            next_state[i] = add_inputs["final_obs"][i]
            if obsfilt is not None:
                next_state[i] = torch.FloatTensor(
                    obsfilt(next_state[i].cpu().numpy(), update=False)
                ).to(self.args.device)

        return self.discrim_net.reward_forward(
            state, action, mask, next_state, self.policy
        )

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument("--airl-reward-shaping", type=str2bool, default=True)
        parser.add_argument(
            "--airl-entropy-bonus",
            type=float,
            default=1.0,
            help="""
                The entropy bonus scaling factor to include in the reward.
            """,
        )
