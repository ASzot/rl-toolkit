from ray import tune
import ray
import rlf.rl.utils as rutils

class RunSettingsTrainable(tune.Trainable):
    def _setup(self, config):
        self.update_i = 0
        run_settings = config['run_settings']
        run_settings.import_add()
        del config['run_settings']
        rutils.update_args(run_settings.args, config, True)
        self.runner = run_settings.setup()
        self.runner.setup()
        if not run_settings.args.ray_debug:
            self.runner.log.disable_print()

    def _train(self):
        updater_log_vals = self.runner.training_iter(self.training_iteration)
        log_dict = self.runner.log_vals(updater_log_vals, self.training_iteration, self.runner.should_log)
        return log_dict




