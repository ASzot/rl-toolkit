import numpy as np

def run_policy(run_settings):
    runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    if run_settings.args.ray:
        from ray import tune
        import ray
        from rlf.rl.ray_runner import RunSettingsTrainable

        # Release resources as they will be recreated by Ray
        runner.close()

        add_config = eval(run_settings.args.ray_config)
        use_config = {'run_settings': run_settings}
        use_config.update(add_config)

        #ray.init(num_cpus=8)
        tune.run(RunSettingsTrainable,
                #resources_per_trial={'cpu': run_settings.args.ray_cpus, "gpu": 0.5},
                stop={'training_iteration': end_update},
                global_checkpoint_period=np.inf,
                config=use_config)
    else:
        args = runner.args

        if runner.should_load_from_checkpoint():
            runner.load_from_checkpoint()

        if args.eval_only:
            return runner.full_eval()

        start_update = 0
        if args.resume:
            start_update = runner.resume()

        runner.setup()
        print('RL Training (%d/%d)' % (start_update, end_update))

        for j in range(start_update, end_update):
            updater_log_vals = runner.training_iter(j)
            if args.log_interval > 0 and (j+1) % args.log_interval == 0:
                log_dict = runner.log_vals(updater_log_vals, j)
            if args.save_interval > 0 and (j+1) % args.save_interval == 0:
                runner.save(j)
            if args.eval_interval > 0 and (j+1) % args.eval_interval == 0:
                runner.eval(j)

        if args.eval_interval > 0:
            runner.eval()
        if args.save_interval > 0:
            runner.save()

        runner.close()
        # WB prefix of the run so we can later fetch the data.
        return args.prefix

