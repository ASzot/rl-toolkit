from rlf.policies.base_policy import BasePolicy


class TabularPolicy(BasePolicy):
    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

    def get_action(self, state, add_state, hxs, masks, step_info):
        return create_np_action_data(0)
