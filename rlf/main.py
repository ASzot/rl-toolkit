# Register some of the env helpers
#import rlf.envs.dm_control_interface
#import rlf.envs.robosuite_interface
#import rlf.envs.fetch_interface
import numpy as np

def run_policy(run_settings):
    if run_settings.args.ray:
        from ray import tune
        import ray
        from rlf.rl.ray_runner import RunSettingsTrainable
        requested_updates = run_settings.get_num_updates()

        add_config = eval(run_settings.args.ray_config)
        use_config = {'run_settings': run_settings}
        use_config.update(add_config)

        #ray.init(num_cpus=8)
        tune.run(RunSettingsTrainable,
                #resources_per_trial={'cpu': run_settings.args.ray_cpus, "gpu": 0.5},
                stop={'training_iteration': requested_updates},
                global_checkpoint_period=np.inf,
                config=use_config)
    else:
        runner = run_settings.setup()
        if runner is None:
            return
        runner.setup()
        runner.full_train()

