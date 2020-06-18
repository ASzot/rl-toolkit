import torch.nn.functional as F
import torch
import gym.spaces as spaces

def td_loss(target, policy, cur_states, cur_actions, add_info, cont_actions=False):
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
    Copy data from `model` to `model_target` with a decay specified by tau
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

