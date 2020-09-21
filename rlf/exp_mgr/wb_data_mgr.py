import sys
sys.path.insert(0, './')

import wandb
from rlf.exp_mgr import config_mgr
from rlf.rl.utils import CacheHelper
import yaml
import argparse
from collections import defaultdict
import pickle
import os
import os.path as osp
import pandas as pd
import hashlib


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    parser.add_argument('--force-refresh', action='store_true', default=False)
    return parser

def get_report_data_from_spec(spec_str, force_refresh=False, cfg='./config.yaml'):
    spec = yaml.safe_load(spec_str)
    return get_report_data(spec['report_name'], spec['plot_column'], spec['fields'],
            force_refresh, cfg)

def get_run_data(run_names, plot_field, method_name,
        cfg='./config.yaml'):
    config_mgr.init(cfg)

    wb_proj_name = config_mgr.get_prop('proj_name')
    wb_entity = config_mgr.get_prop('wb_entity')

    all_df = None

    api = wandb.Api()
    for run_name in run_names:
        runs = api.runs(f"{wb_entity}/{wb_proj_name}", {"config.prefix": run_name})
        assert len(runs) == 1
        wbrun = next(iter(runs))
        df = wbrun.history(samples=15000)
        df = df[['_step', plot_field]]
        df['run'] = run_name

        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])

    all_df['method'] = method_name

    return all_df

def get_run_ids_from_report(wb_entity, wb_proj_name, report_name, get_sections, api):
    reports = api.reports(wb_entity + '/' + wb_proj_name)
    report = None
    for cur_report in reports:
        id_parts = cur_report.description.split('ID:')
        if len(id_parts) > 1:
            cur_id = id_parts[1].split(' ')[0]
            if report_name == cur_id:
                report = cur_report
                break
    if report is None:
        raise ValueError('Could not find report')

    # Find which section the run sets are in
    report_section_idx = None
    for i in range(len(report.sections)):
        if 'runSets' in report.sections[i]:
            report_section_idx = i
            break

    run_ids = []
    for run_set in report.sections[report_section_idx]['runSets']:
        report_section = run_set['name']
        if report_section not in get_sections:
            continue
        report_runs = run_set['selections']['tree']
        for run_id in report_runs:
            run_ids.append((report_section, run_id))
    return run_ids

def get_report_data(report_name, plot_field, plot_sections,
        force_refresh=False, cfg='./config.yaml'):
    """
    Converts the selected data sets in a W&B report into a Pandas DataFrame.
    Fetches only the plot_field you specify.
    """
    config_mgr.init(cfg)

    wb_proj_name = config_mgr.get_prop('proj_name')
    wb_entity = config_mgr.get_prop('wb_entity')

    save_report_name = report_name.replace(' ', '-').replace("/", "-")
    cacher = CacheHelper(f"{wb_entity}_{wb_proj_name}_{save_report_name}",
            plot_sections)
    all_df = None

    if cacher.exists() and not force_refresh:
        all_df = cacher.load()
        uniq_methods = all_df['method'].unique()
        for k in uniq_methods:
            idx = plot_sections.index(k)
            del plot_sections[idx]
        if len(plot_sections) == 0:
            return all_df

    api = wandb.Api()
    run_ids = get_run_ids_from_report(wb_entity, wb_proj_name, report_name, plot_sections, api)
    for report_section, run_id in run_ids:
        wbrun = api.run(f"{wb_entity}/{wb_proj_name}/{run_id}")
        df = wbrun.history(samples=15000)
        df = df[['_step', plot_field]]
        df['method'] = report_section
        df['run'] = run_id

        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])

    uniq_methods = all_df['method'].unique()
    for plot_section in plot_sections:
        assert plot_section in uniq_methods, f"{plot_section} not found"

    cacher.save(all_df)

    return all_df

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    #df = get_run_data([
    #    '52-MGFR-31-KY-dpf',
    #    '51-MGFR-51-GU-dpf',
    #    '51-MGFR-41-HE-dpf',
    #    ], 'avg_ep_found_goal',
    #    'Ours', args.cfg)

    df = get_report_data(
            "gw-final-locator",
            "avg_ep_found_goal",
            ['ours'],
            args.force_refresh,
            args.cfg
            )
    print(df.head())

    #raw_dat = get_report_data_from_spec('''
    #        report_name: "5/2/20 Grid World Analysis"
    #        plot_column: "avg_r"
    #        fields:
    #        - "ours (0.9 cover, 100%)"
    #        - "ours (0.9 cover, 100%, ablate)"
    #        ''', args.force_refresh, args.cfg)



