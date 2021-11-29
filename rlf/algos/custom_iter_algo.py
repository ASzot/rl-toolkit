class CustomIterAlgo:
    """
    Overrides the training step with completely custom logic. This is useful if
    we want to collect data in a special way for example performing multiple
    rollouts in one overall update. An example of this is meta-learning where
    we need to track updates across multiple separate rollouts and then make
    one overall update to whatever is being meta-learned.

    Any class that derives will have training_iter called, see class `Runner` for more.
    """

    def training_iter(self, rollout_fn, update_iter: int):
        """
        :param rollout_fn: For the definition of rollout_fn see `Runner::rl_rollout`. This rolls out a policy to collect a batch of experience from the environment.
        """
        raise NotImplementedError()
