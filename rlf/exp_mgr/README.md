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
