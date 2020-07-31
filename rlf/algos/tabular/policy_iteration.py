from rlf.algos.tabular.base_tabular import BaseTabular

class PolicyIteration(BaseTabular):
    def init(self, policy, args):
        assert isinstance(policy, ActionValuePolicy)
        super().init(policy, args)

    def update(self, storage):
        return {}

    def get_add_args(self, parser):
        super().get_add_args(parser)
        #########################################
        # Overrides
        # No environment interaction
        parser.add_argument('--num-steps', type=int, default=0)

        #########################################
        # New args
        parser.add_argument('--lr', type=float, default=None)
