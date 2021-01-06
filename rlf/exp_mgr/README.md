## Running Jobs
### SLURM Helper
The SLURM helper is only activated when you are running `--sess-id X` and
specify a value for `--st`. When `habitat_baselines` is in the command a
`sh` file will be created with the command and then `sbatch` will be run.
`--nodes` does not affect non-sbatch runs. 

- `--cd -1` Does not set the CUDA environment variable. This is helpful on
  machines where you shouldn't mess with this setting. 

## Setting up W&B
- First log into W&B: `wandb login ...`


## Getting Data From W&B
- To get data from a particular run (where you know the name of the run) use
  `get_run_data`. You can specify a list of runs you want to get data for. 
- To get data for data sources in a report you can use `get_report_data`. To
  make this work you need to specify the name of the report in the
  *description*. So if you are looking up report with name "my-report" the
  description has to contain "ID:my-report".

## config.yaml
The settings that need to go into `config.yaml` are:
- `cmds_loc`
- `wb_proj_name`

## Plotting 
To plot, use `auto_plot.py`. Typically this might be run as `python rl-toolkit/rlf/exp_mgr/auto_plot.py --plot-cfg my_plot_cfgs/plot.yaml`. Here is an illustrative plot settings yaml file. 

```
---
plot_sections:
    - plot_title: "Fridge"
      save_name: "pick_fridge"
      report_name: "0106-pick-fridge"
      y_bounds: "0.0,100.0"
      x_disp_bounds: "0,60e6"
      y_disp_bounds: "0,110"
      legend: True
      line_sections:
          - "mpg"
          - "blind"
          - "mpg margin"
          - "mpg obj"
      plot_sections:
          - "img old"
          - "state old"
      force_reload: False
global_renames: 
    "eval_metrics/ep_success": "Success (%)"
    _step: "Step"
    'img old': 'Image'
    'state old': 'State Only'
    'img': 'Image'
    'state': 'State'
    "blind": 'Blind'
    "mpg": 'MP+Geom'
    "mpg margin": 'MP+Geom+Margin'
    "mpg obj": 'MP+Geom+Obj'
plot_key: "eval_metrics/ep_success"
line_plot_key: "eval_metrics/ep_success"
smooth_factor: 0.95
scale_factor: 100
line_op: 'max'
line_val_key: "eval_metrics/ep_success"
line_plot_key: "eval_metrics/ep_success"
config_yaml: "./config.yaml"
save_loc: "./data/plot/figs/"
fig_dims: [6,4]
legend_font_size: 'medium'
name_match_pat: ['_eval', 'mpg', 'blind']
colors:
    "img old": 0 
    "state old": 1
    "img": 0 
    "state": 1
    "blind": 2
    "mpg": 3
    "mpg margin": 4
    "mpg obj": 5
```

