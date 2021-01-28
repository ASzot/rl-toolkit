import sys
sys.path.insert(0, './')
import argparse
import yaml
from rlf.rl.utils import human_format_int
from rlf.exp_mgr.wb_data_mgr import get_report_data
from rlf.exp_mgr.plotter import uncert_plot, high_res_save, MARKER_ORDER
import matplotlib.pyplot as plt
import os.path as osp
import os
import pandas as pd
import numpy as np
import seaborn as sns

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-cfg', type=str, required=True)
    parser.add_argument('--legend', action='store_true')
    return parser


def export_legend(ax, line_width, filename="legend.pdf"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False,
            loc='lower center', ncol=10, handlelength=2)
    for line in legend.get_lines():
        line.set_linewidth(line_width)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def plot_legend(plot_cfg_path):
    with open(plot_cfg_path) as f:
        plot_settings = yaml.load(f)
        colors = sns.color_palette()
        group_colors = {name: colors[idx] for name, idx in
                plot_settings['colors'].items()}

        for section_name, section in plot_settings['plot_sections'].items():
            fig, ax = plt.subplots(figsize=(5, 4))
            names = section.split(',')
            darkness = plot_settings['marker_darkness']
            for name in names:
                add_kwargs = {}
                if name in plot_settings['linestyles']:
                    linestyle = plot_settings['linestyles'][name]
                    if isinstance(linestyle, list):
                        add_kwargs['linestyle'] = linestyle[0]
                        add_kwargs['dashes'] = linestyle[1]
                    else:
                        add_kwargs['linestyle'] = linestyle

                disp_name = plot_settings['name_map'][name]
                midx = plot_settings['colors'][name] % len(MARKER_ORDER)
                ax.plot([0], [1], marker=MARKER_ORDER[midx], label=disp_name,
                        color=group_colors[name],
                        markersize=plot_settings['marker_size'],
                        markeredgewidth=plot_settings['marker_width'],
                        markeredgecolor=(darkness, darkness, darkness, 1),
                        **add_kwargs)
            export_legend(ax, plot_settings['line_width'],
                    osp.join(plot_settings['save_loc'], section_name + '_legend.pdf'))
            plt.clf()

def plot_from_file(plot_cfg_path):
    with open(plot_cfg_path) as f:
        plot_settings = yaml.load(f)

        colors = sns.color_palette()
        group_colors = {name: colors[idx] for name, idx in
                plot_settings['colors'].items()}

        def get_setting(local, k, local_override=True):
            if local_override:
                if k in local:
                    return local[k]
                else:
                    return plot_settings[k]
            else:
                if k in plot_settings:
                    return plot_settings[k]
                else:
                    return local[k]

        for plot_section in plot_settings['plot_sections']:
            plot_key = plot_section.get('plot_key', plot_settings['plot_key'])
            match_pat = plot_section.get('name_match_pat',
                    plot_settings.get('name_match_pat', None))
            print(f"Getting data for {plot_section['report_name']}")
            plot_df = get_report_data(plot_section['report_name'],
                    plot_key,
                    plot_section['plot_sections'],
                    get_setting(plot_section,'force_reload', False),
                    match_pat,
                    plot_settings.get('other_plot_keys', []),
                    plot_settings['config_yaml'])

            if 'line_sections' in plot_section:
                line_plot_key = get_setting(plot_section, 'line_plot_key')
                take_operation = get_setting(plot_section, 'line_op')
                line_val_key = get_setting(plot_section, 'line_val_key')
                if line_plot_key != line_val_key:
                    fetch_keys = [line_plot_key, line_val_key]
                else:
                    fetch_keys = line_plot_key
                line_df = get_report_data(plot_section['report_name'],
                        fetch_keys,
                        plot_section['line_sections'],
                        get_setting(plot_section,'force_reload', False),
                        match_pat,
                        plot_settings['config_yaml'])
                line_df = line_df[line_df[line_plot_key].notna()]
                uniq_step = plot_df['_step'].unique()
                use_line_df = None
                for group_name, df in line_df.groupby('run'):
                    #df = df.iloc[np.array([0]).repeat(len(uniq_step))]
                    if take_operation == 'min':
                        use_idx = np.argmin(df[line_val_key])
                    elif take_operation == 'max':
                        use_idx = np.argmax(df[line_val_key])
                    elif take_operation == 'final':
                        use_idx = -1
                    else:
                        raise ValueError('Unrecognized line reduce')
                    df = df.iloc[np.array([use_idx]).repeat(len(uniq_step))]
                    if line_plot_key != line_val_key:
                        del df[line_val_key]

                    df.index = np.arange(len(uniq_step))
                    df['_step'] = uniq_step
                    if use_line_df is None:
                        use_line_df = df
                    else:
                        use_line_df = pd.concat([use_line_df, df])
                use_line_df = use_line_df.rename(columns={line_plot_key: plot_key})
                plot_df = pd.concat([plot_df, use_line_df])

            use_fig_dims = plot_section.get('fig_dims', plot_settings.get('fig_dims', (5,4)))
            fig, ax = plt.subplots(figsize=use_fig_dims)
            def get_nums_from_str(s):
                return [float(x) for x in s.split(',')]

            local_renames = {}
            if 'renames' in plot_section:
                local_renames = plot_section['renames']

            title = plot_section['plot_title']
            if 'scale_factor' in plot_settings:
                plot_df[plot_key] *= plot_settings['scale_factor']
            use_legend_font_size = plot_section.get('legend_font_size',
                    plot_settings.get('legend_font_size', 'x-large'))
            uncert_plot(plot_df, ax, '_step', plot_key, 'run', 'method',
                    get_setting(plot_section, 'smooth_factor'),
                    y_bounds=get_nums_from_str(plot_section['y_bounds']),
                    x_disp_bounds=get_nums_from_str(plot_section['x_disp_bounds']),
                    y_disp_bounds=get_nums_from_str(plot_section['y_disp_bounds']),
                    xtick_fn=human_format_int,
                    legend=plot_section['legend'],
                    title=title,
                    group_colors=group_colors,
                    method_idxs=plot_settings['colors'],
                    tight=True,
                    legend_font_size=use_legend_font_size,
                    num_marker_points=plot_settings.get('num_marker_points', {}),
                    line_styles=plot_settings.get('linestyles', {}),
                    rename_map={
                        **plot_settings['global_renames'],
                        **local_renames,
                        })
            save_loc = plot_settings['save_loc']
            if not osp.exists(save_loc):
                os.makedirs(save_loc)
            save_path = osp.join(save_loc, plot_section['save_name'] + '.pdf')
            high_res_save(save_path)
            plt.clf()

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.legend:
        plot_legend(args.plot_cfg)
    else:
        plot_from_file(args.plot_cfg)
