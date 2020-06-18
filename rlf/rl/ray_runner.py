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
        self.args = run_settings.args

        if not run_settings.args.ray_debug:
            self.runner.log.disable_print()

    def _train(self):
        updater_log_vals = self.runner.training_iter(self.training_iteration)
        if (self.training_iteration+1) % self.args.log_interval == 0:
            log_dict = self.runner.log_vals(updater_log_vals, self.training_iteration)
        if (self.training_iteration+1) % self.args.save_interval == 0:
            self.runner.save()
        if (self.training_iteration+1) % self.args.eval_interval == 0:
            self.runner.eval()

        return log_dict




