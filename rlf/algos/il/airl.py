import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.gaifo import GaifoDiscrim
from rlf.algos.il.gail import GailDiscrim
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO


class AIRL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([AirlDiscrim(get_discrim), agent_updater], 1)


class AirlNetDiscrim(nn.Module):
    """
    The discriminator network is from https://github.com/ku2482/gail-airl-ppo.pytorch/blob/master/gail_airl_ppo/network/disc.py
    """

    def __init__(self, state_enc, gamma, hidden_dim=64):
        super().__init__()
        self.state_enc = state_enc.net
        output_size = state_enc.output_shape[0]
        self.g = nn.Sequential(
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.h = nn.Sequential(
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gamma = gamma

    def f(self, states, mask, next_states):
        states_enc = self.state_enc(states)
        next_states_enc = self.state_enc(next_states)
        rs = self.g(states_enc)
        vs = self.h(states_enc)
        next_vs = self.h(next_states_enc)
        return rs + self.gamma * mask * next_vs - vs

    def forward(self, states, actions, mask, next_states, policy):
        with torch.no_grad():
            log_pis = policy.evaluate_actions(states, {}, {}, mask, actions)["log_prob"]
        return self.f(states, mask, next_states) - log_pis


class AirlDiscrim(GailDiscrim):
    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        base_net = self.policy.get_base_net_fn(ob_shape)

        return AirlNetDiscrim(
            base_net, self.args.gamma, self.args.gail_disc_hidden_dim
        ).to(self.args.device)

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

        expert_d = self.discrim_net(
            exp_s0, expert_actions, expert_mask, exp_s1, self.policy
        )
        agent_d = self.discrim_net(
            agent_s0, agent_actions, agent_mask, agent_s1, self.policy
        )
        return expert_d, agent_d, 0

    def _compute_disc_val(self, state, next_state, action):
        raise NotImplementedError("Not used in AIRL")

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        settings.include_info_keys.extend(
            [("final_obs", lambda env: rutils.get_obs_shape(env.observation_space))]
        )
        return settings

    def _compute_discrim_reward(self, state, next_state, action, mask, add_inputs):
        finished_episodes = [i for i in range(len(mask)) if mask[i] == 0.0]
        obsfilt = self.get_env_ob_filt()
        for i in finished_episodes:
            next_state[i] = add_inputs["final_obs"][i]
            if obsfilt is not None:
                next_state[i] = torch.FloatTensor(
                    obsfilt(next_state[i].cpu().numpy(), update=False)
                ).to(self.args.device)

        logits = self.discrim_net(state, action, mask, next_state, self.policy)
        return -F.logsigmoid(-logits)
