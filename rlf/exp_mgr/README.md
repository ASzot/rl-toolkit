## Running Jobs
### SLURM Helper
The SLURM helper is only activated when you are running `--sess-id X` and
specify a value for `--st`. When `habitat_baselines` is in the command a
`sh` file will be created with the command and then `sbatch` will be run.
`--nodes` does not affect non-sbatch runs. 

- `--cd -1` does not set the CUDA environment variable (which is the default). This is helpful on
  machines where you shouldn't mess with this setting. 

## Setting up W&B
- First log into W&B: `wandb login ...`


## Getting Data From W&B
1. To get data from a particular run (where you know the name of the run) use
  `get_run_data`. You can specify a list of runs you want to get data for. 
2. To get data for data sources in a report you can use `get_report_data`. When accessing reports, you need look up the report by **description, not name**. So if you want to get a report called "my-report" from the code, the description of the report on W&B should be "ID:my-report". 

## config.yaml
The settings that need to go into `config.yaml` are:
- `cmds_loc`
- `wb_proj_name`

## Plotting 
To plot, use `auto_plot.py`. This will automatically fetch and plot runs from
reports on W&B. It has support for plotting horizontal lines, specifying the
color, axes, and which key to plot. **The report on W&B has to follow [this
naming convention from point
2](https://github.com/ASzot/rl-toolkit/tree/master/rlf/exp_mgr#getting-data-from-wb)**.
Typically this is run as `python rl-toolkit/rlf/exp_mgr/auto_plot.py --plot-cfg
my_plot_cfgs/plot.yaml`. 

Here is an illustrative plot settings YAML file. 

```
---
# You can specify multiple elements in the plot  sections list to create multiple plots at once. 
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
      plot_sections:
          - "img"
          - "state"
      force_reload: False
global_renames: 
    "eval_metrics/ep_success": "Success (%)"
    _step: "Step"
    'img': 'Image Method'
    'state': 'State Only Method'
    "mpg": 'Motion Planning Method'
plot_key: "eval_metrics/ep_success"
line_plot_key: "eval_metrics/ep_success"
smooth_factor: 0.95
# Multiplies the Y axis values by this amount. Helpful if you want to turn
# something into a percen.
scale_factor: 100
# These configure how a run should be converted into a line (single value). 
line_op: 'max'
line_val_key: "eval_metrics/ep_success"
line_plot_key: "eval_metrics/ep_success"
config_yaml: "./config.yaml"
save_loc: "./data/plot/figs/"
# Make the figure wider, this is optional. 
fig_dims: [6,4]
# Make legend font size smaller. 
legend_font_size: 'medium'
# This will ignore names from W&B which don't contain one of these substrings.
name_match_pat: ['_eval', 'mpg', 'blind']
colors:
    "img": 0 
    "state": 1
    "mpg": 2
```

There is also a utility for creating a separate PDF file containing just the
legend. This is run as `python rl-toolkit/rlf/exp_mgr/auto_plot.py --plot-cfg
my_plot_cfgs/my_legend.yaml` --legend. An example of a legend YAML file is
below. The marker attributes refer to the characteristics of the line next to
the name. 

```
---
plot_sections:
    ablate: "ours,gaifo,gaifo-s"
save_loc: "./data/plot/figs/final"
marker_size: 12
marker_width: 0.0
marker_darkness: 0.1
line_width: 3.0
name_map:
    ours: "Ours"
    gaifo: "GAIfO"
    gail-s: "GAIfO-s"
colors:
    ours: 0 
    gaifo: 1
    gail-s: 2
```
