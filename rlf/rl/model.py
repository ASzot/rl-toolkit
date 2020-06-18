import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def def_mlp_weight_init(m):
    return weight_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNet(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_shape(self):
        return (self._hidden_size,)

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class IdentityBase(BaseNet):
    def __init__(self, input_shape):
        super().__init__(False, None, None)
        self.input_shape = input_shape

    def net(self, x):
        return x

    @property
    def output_shape(self):
        return self.input_shape

    def forward(self, inputs, rnn_hxs, masks):
        return inputs, None


class CNNBase(BaseNet):
    def __init__(self, num_inputs, recurrent, hidden_size):
        super().__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: weight_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.net = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.net(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return x, rnn_hxs

class MLPBase(BaseNet):
    def __init__(self, num_inputs, recurrent, hidden_sizes,
            weight_init=def_mlp_weight_init):
        super().__init__(recurrent, num_inputs, hidden_sizes[-1])

        assert len(hidden_sizes) > 0

        layers = [weight_init(nn.Linear(num_inputs, hidden_sizes[0])), nn.Tanh()]
        # Minus one for the input layer
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                    weight_init(nn.Linear(hidden_sizes[i], hidden_sizes[i+1])),
                    nn.Tanh()
                ])

        self.net = nn.Sequential(*layers)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.net(x)

        return hidden_actor, rnn_hxs


class MLPBasic(MLPBase):
    def __init__(self, num_inputs, hidden_size, num_layers,
            weight_init=def_mlp_weight_init):
        super().__init__(num_inputs, False, [hidden_size] * num_layers,
                weight_init)

class TwoLayerMlpWithAction(BaseNet):
    def __init__(self, num_inputs, hidden_sizes, action_dim):
        assert len(hidden_sizes) == 2, 'Only two hidden sizes'
        super().__init__(False, num_inputs, hidden_sizes[-1])

        self.fc1 = def_mlp_weight_init(nn.Linear(num_inputs, hidden_sizes[0]))
        self.fc2 = def_mlp_weight_init(nn.Linear(hidden_sizes[0] + action_dim, hidden_sizes[1]))

        self.train()

    def forward(self, inputs, actions, rnn_hxs, masks):
        x = F.tanh(self.fc1(inputs))

        x = F.tanh(self.fc2(torch.cat([x, actions], dim=-1)))
        return x, rnn_hxs

class InjectNet(nn.Module):
    def __init__(self, base_net, head_net, base_net_out_dim,
            head_net_in_dim, inject_dim, should_inject):
        super().__init__()
        self.base_net = base_net
        if not should_inject:
            inject_dim = 0
        self.inject_layer = nn.Sequential(
                nn.Linear(base_net_out_dim + inject_dim, head_net_in_dim),
                nn.Tanh())
        self.head_net = head_net
        self.should_inject = should_inject

    def forward(self, x, inject_x):
        x = self.base_net(x)
        if self.should_inject:
            x = torch.cat([x, inject_x], dim=-1)
        x = self.inject_layer(x)
        x = self.head_net(x)
        return x





