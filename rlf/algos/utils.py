import torch.nn.functional as F
import torch
import gym.spaces as spaces
from torch import autograd
import numpy as np

def linear_lr_schedule(cur_update, total_updates, initial_lr, opt):
    lr = initial_lr - \
        (initial_lr * (cur_update / float(total_updates)))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def td_loss(target, policy, cur_states, cur_actions, add_info={}, cont_actions=False):
    """
    Computes the mean squared error between the Q values for the current states
    and the target q values.
    """
    if cont_actions:
        inputs = torch.cat([cur_states, cur_actions], dim=-1)
        cur_q_vals = policy.get_value(inputs, **add_info)
    else:
        cur_q_vals = policy(cur_states, **add_info).gather(1, cur_actions)
    loss = F.mse_loss(cur_q_vals.view(-1), target.view(-1))
    return loss

def soft_update(model, model_target, tau):
    """
    Copy data from `model` to `model_target` with a decay specified by tau. A
    tau value closer to 0 means less of the model will be copied to the target
    model.
    """
    for param, target_param in zip(model.parameters(), model_target.parameters()):
        target_param.detach()
        target_param.data.copy_((tau * param.data) + ((1.0 - tau) * target_param.data))

def hard_update(model, model_target):
    """
    Copy all data from `model` to `model_target`
    """
    model_target.load_state_dict(model.state_dict())


def reparam_sample(dist):
    """
    A general method for updating either a categorical or normal distribution.
    In the case of a Categorical distribution, the logits are just returned
    """
    if isinstance(dist, torch.distributions.Normal):
        return dist.rsample()
    elif isinstance(dist, torch.distributions.Categorical):
        return dist.logits
    else:
        raise ValueError('Unrecognized distribution')


def compute_ac_loss(pred_actions, true_actions, ac_space):
    if isinstance(pred_actions, torch.distributions.Distribution):
        pred_actions = reparam_sample(pred_actions)

    if isinstance(ac_space, spaces.Discrete):
        loss = F.cross_entropy(pred_actions, true_actions.view(-1).long())
    else:
        loss = F.mse_loss(pred_actions, true_actions)
    return loss



# Adapted from https://github.com/Khrylx/PyTorch-RL/blob/f44b4444c9db5c1562c5d0bc04080c319ba9141a/utils/torch.py#L26
def set_flat_params_to(params, flat_params):
    prev_ind = 0
    for param in params:
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


# Adapted from https://github.com/Khrylx/PyTorch-RL/blob/f44b4444c9db5c1562c5d0bc04080c319ba9141a/utils/torch.py#L17
def get_flat_params_from(params):
    return torch.cat([param.view(-1) for param in params])


def wass_grad_pen(expert_state, expert_action, policy_state, policy_action,
        use_actions, disc_fn):
    num_dims = len(expert_state.shape) - 1
    alpha = torch.rand(expert_state.size(0), 1)
    alpha_state = alpha.view(-1, *[1 for _ in range(num_dims)]
                             ).expand_as(expert_state).to(expert_state.device)
    mixup_data_state = alpha_state * expert_state + \
        (1 - alpha_state) * policy_state
    mixup_data_state.requires_grad = True

    alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
    mixup_data_action = alpha_action * expert_action + \
        (1 - alpha_action) * policy_action
    mixup_data_action.requires_grad = True

    disc = disc_fn(mixup_data_state, mixup_data_action)
    ones = torch.ones(disc.size()).to(disc.device)

    inputs = [mixup_data_state]
    if use_actions:
        inputs.append(mixup_data_action)

    grad = autograd.grad(
        outputs=disc,
        inputs=inputs,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
    return grad_pen

class StackHelper:
    """
    A helper for stacking observations.
    """

    def __init__(self, ob_shape, n_stack, device, n_procs=None):
        self.input_dim = ob_shape[0]
        self.n_procs = n_procs
        self.real_shape = (n_stack*self.input_dim, *ob_shape[1:])
        if self.n_procs is not None:
            self.stacked_obs = torch.zeros((n_procs, *self.real_shape))
            if device is not None:
                self.stacked_obs = self.stacked_obs.to(device)
        else:
            self.stacked_obs = np.zeros(self.real_shape)

    def update_obs(self, obs, dones=None, infos=None):
        """
        - obs: torch.tensor
        """
        if self.n_procs is not None:
            self.stacked_obs[:, :-self.input_dim] = self.stacked_obs[:, self.input_dim:].clone()
            for (i, new) in enumerate(dones):
                if new:
                    self.stacked_obs[i] = 0
            self.stacked_obs[:, -self.input_dim:] = obs

            # Update info so the final observation frame stack has the final
            # observation as the final frame in the stack.
            for i in range(len(infos)):
                if 'final_obs' in infos[i]:
                    new_final = torch.zeros(*self.stacked_obs.shape[1:])
                    new_final[:-1] = self.stacked_obs[i][1:]
                    new_final[-1] = torch.tensor(infos[i]['final_obs']).to(self.stacked_obs.device)
                    infos[i]['final_obs'] = new_final
            return self.stacked_obs.clone(), infos
        else:
            self.stacked_obs[:-self.input_dim] = self.stacked_obs[self.input_dim:].copy()
            self.stacked_obs[-self.input_dim:] = obs

            return self.stacked_obs.copy(), infos

    def reset(self, obs):
        if self.n_procs is not None:
            if torch.backends.cudnn.deterministic:
                self.stacked_obs = torch.zeros(self.stacked_obs.shape)
            else:
                self.stacked_obs.zero_()
            self.stacked_obs[:, -self.input_dim:] = obs
            return self.stacked_obs.clone()
        else:
            self.stacked_obs = np.zeros(self.stacked_obs.shape)
            self.stacked_obs[-self.input_dim:] = obs
            return self.stacked_obs.copy()

    def get_shape(self):
        return self.real_shape

