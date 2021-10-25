## Running Jobs
Argument list
* `--cd X` A value of -1 does not set the CUDA environment variable (which is the default). This is helpful on
  machines where you shouldn't mess with this setting. 
* `--skip-env` Skips adding the environment variables under `add_env_vars` from
    config.
* `--skip-add` Skips adding the command line arguments from `change_cmds` from
    config.
    

### SLURM Helper
The SLURM helper is only activated when you are running `--sess-id X` and
specify a value for `--st`. 


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

You can also create bar plots the exact same way with `python rl-toolkit/rlf/exp_mgr/auto_bar.py --plot-cfg my_plot_cfgs/plot.yaml`. 
Bar plots only work with `line_sections` and do not work with `plot_sections`,
but otherwise use almost exactly the same configuration format. 

Documentation of the plot settings YAML file. Most properties in the
`plot_sections` element can also be set in the base element as a default. 
```
---
plot_sections:
    - save_name: [string, filename to save to (not including extension or directory)]
      should_save: [bool, If False, then the plot is not saved, and is apart of the next rendered plot, this is how you can compose data from multiple data sources.]

      # Data source settings
      report_name: [string, either the TB search directory or the W&B report description ID]
      line_report_name: [string, W&B ID or TB directory of where to get data for rendering lines, if nothing, then same as report name]

      # Render settings
      plot_title: [string, overhead title of plot]
      y_bounds: [string, like "0.0,100.0" clips values to this range]
      x_disp_bounds: [string, like "0,1e8" plot display bounds]
      y_disp_bounds: [string, like "60,102" plot display bounds ]
      legend: [bool, If true, will render the legend within the plot]
      nlegend_cols: [int, # of columns in the legend, useful for large legends]

      make_steps_similar: [bool, if True, will forcefully align the steps in the runs for a particular method]
      plot_key: "eval_metrics/ep_success"

      line_match_pat: null
      line_sections:
          - 'mpg'
          - 'mpp2'
      plot_sections:
          - "D-s eval"
          - "ours"
          - "RGBD eval"
      force_reload: False
      # Optional, if provided this will linearly scale the desired key of a particular plot_section 
      # Notice we can scale the x-axis by scaling "_step". However, scaling any
      # key is possible. Supports default in outer config
      scaling:
        "ours": 
          offset: 4096
          scale: 128
          scale_key: '_step'
force_reload: True # This force reload overrides the local ones.
global_renames: [dict str -> str]
    # You can use the y-axis values!
    "eval_metrics/ep_success": "Success (%)"
    # Or the x-axis values
    _step: "Step"

    # Method names
    "D-s eval": "D+ps"
line_plot_key: "eval_metrics/ep_success"
line_val_key: "eval_metrics/ep_success"
smooth_factor: 0.8
scale_factor: 100
line_op: 'max'
config_yaml: "./config.yaml"
save_loc: "./data/plot"
fig_dims: [6,4] # The save dimensions of the figure, by default this is [5,4]
legend_font_size: 'medium'
linestyles: [dict str -> str, the value is the matplotlib line style] 
colors:
  "RGBD_s": 0
  "input_ablation_RGB_s": 1
  "input_ablation_D_s": 2
  "input_ablation_s": 3
  "input_ablation_RGBD_g": 4
  "input_ablation_RGB_g": 5
  "input_ablation_D_g": 6
  "mpg": 7
  "mpp": 8
  "mpp2": 8

  "D-s eval": 0
  "D eval": 1
  "RGB-s eval": 2
  "RGB eval": 3
  "RGBD-s eval": 4
  "RGBD eval": 5
```

