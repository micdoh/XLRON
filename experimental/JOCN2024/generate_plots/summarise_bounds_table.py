import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
from scipy import interpolate

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot_style import configure_style, increase_legend_line_thickness


def load_bounds_file(path):
    """Load a bounds results JSONL file (produced by --DATA_OUTPUT_FILE).

    Flattens the nested JSON structure into a flat DataFrame with columns:
    experiment, topology, load, k, heur, plus metric columns
    (e.g. service_blocking_probability_mean).

    Returns a pandas DataFrame.
    """
    path = Path(path)
    jsonl_path = path.with_suffix('.jsonl') if path.suffix != '.jsonl' else path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"No bounds data file found at {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = {}
            config = obj.get('config', {})
            row['experiment'] = config.get('PROJECT', '')
            row['topology'] = config.get('topology_name', '')
            row['load'] = config.get('load', 0)
            row['k'] = config.get('k', 0)
            row['heur'] = config.get('path_heuristic', '')
            row['link_resources'] = config.get('link_resources', 0)
            # Flatten metrics: {metric_name: {stat: val}} -> metric_stat columns
            for metric_name, stats in obj.get('metrics', {}).items():
                for stat_name, stat_val in stats.items():
                    row[f'{metric_name}_{stat_name}'] = stat_val
            records.append(row)
    return pd.DataFrame(records)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def find_x_at_y(x_data, y_data, target_y=0.1):
    """
    Find x value where y crosses target_y using linear interpolation.

    Parameters:
    x_data (array-like): x-axis values
    y_data (array-like): y-axis values
    target_y (float): y value to find crossing point (default 0.1)

    Returns:
    float: x value where y crosses target_y, or None if no crossing found
    """
    # Create interpolation function
    f = interpolate.interp1d(y_data, x_data, bounds_error=False)

    # Find x value at target_y
    x_at_y = f(target_y)

    # Check if we found a valid crossing point
    if np.isnan(x_at_y):
        return None

    return float(x_at_y)


def add_gap_line(ax, pub, topology, n_slots, heur_data, bounds_data, y_value=0.1, color='red',
                 linewidth=2, alpha=0.7, cap_size=10):
    """
    Add horizontal line showing gap between heuristic and bounds at specified y value.

    Parameters:
    ax: matplotlib axis
    pub: publication name
    topology: topology name
    n_slots: number of slots
    heur_data: DataFrame with heuristic data
    bounds_data: DataFrame with bounds data
    y_value: y value to show gap at (default 0.1)
    color: line color (default 'red')
    linewidth: line width (default 2)
    alpha: line transparency (default 0.7)

    Returns:
    bool: True if line was added successfully
    """
    # Filter data for this case
    pub_filter = 'PtrNet-RSA' if 'PtrNet-RSA' in pub else pub
    case_heur = heur_data[(heur_data['publication'] == pub_filter) &
                          (heur_data['topology'] == topology) &
                          (heur_data['N_slots'] == n_slots)]
    case_bounds = bounds_data[(bounds_data['publication'] == pub_filter) &
                              (bounds_data['topology'] == topology) &
                              (bounds_data['N_slots'] == n_slots)]

    if case_heur.empty or case_bounds.empty:
        return False

    # Find x values at target y
    x_heur = find_x_at_y(case_heur['load'], case_heur['mean'], y_value)
    x_bounds = find_x_at_y(case_bounds['load'], case_bounds['mean'], y_value)

    if x_heur is None or x_bounds is None:
        return False

    # Plot horizontal line
    ax.hlines(y=y_value, xmin=x_bounds, xmax=x_heur,
              color=color, linewidth=linewidth, alpha=alpha)
    ax.vlines([x_bounds, x_heur], y_value - cap_size / 1000, y_value + cap_size / 1000,
              color=color, linewidth=linewidth, alpha=alpha)

    # Add text showing gap size
    gap = 100 * (x_bounds - x_heur) / x_heur
    ax.text((x_bounds + x_heur) / 2, y_value * 1.2,
            f'{gap:.0f}%',
            horizontalalignment='center',
            color=color)

    return True


# Modify your plot_case function to include the gap line
def plot_case(ax, pub, topology, n_slots, heur_data, bounds_data):
    pub_filter = 'PtrNet-RSA' if 'PtrNet-RSA' in pub else pub
    case_heur = heur_data[(heur_data['publication'] == pub_filter) &
                          (heur_data['topology'] == topology) &
                          (heur_data['N_slots'] == n_slots)]
    case_bounds = bounds_data[(bounds_data['publication'] == pub_filter) &
                              (bounds_data['topology'] == topology) &
                              (bounds_data['N_slots'] == n_slots)]

    lines = []
    labels = []

    if not case_heur.empty:
        line = ax.plot(case_heur['load'], case_heur['mean'], label='Heuristic',
                       marker='o', markerfacecolor=heur_col, linestyle='-', color=heur_col)
        ax.fill_between(case_heur['load'], case_heur['band_lower'],
                        case_heur['band_upper'], alpha=0.2, color=heur_col)
        lines.append(line[0])
        labels.append('Best heuristic')

    if not case_bounds.empty:
        line = ax.plot(case_bounds['load'], case_bounds['mean'], label='Bounds',
                       marker='o', markerfacecolor=bounds_col, linestyle='-', color=bounds_col)
        ax.fill_between(case_bounds['load'], case_bounds['band_lower'],
                        case_bounds['band_upper'], alpha=0.2, color=bounds_col)
        lines.append(line[0])
        labels.append('Defragmentation bound')

    # Add gap line at 0.1%
    add_gap_line(ax, pub, topology, n_slots, heur_data, bounds_data)

    pub = 'Deep/Reward/GCN-RMSA' if pub == 'DeepRMSA~Reward-RMSA~GCN-RMSA' else pub

    title = f'{pub}\n\n{topology}' if topology == 'NSFNET' else topology
    ax.set_title(title, fontsize=32)
    return True, lines, labels

if __name__ == '__main__':
    configure_style()

    data_dir = Path(__file__).resolve().parents[1] / 'results' / 'bounds'
    heur_data = load_bounds_file(data_dir / 'experiment_results_eval_bounds')
    bounds_data = load_bounds_file(data_dir / 'experiment_results_reconfigurable_bounds')
    _rename = {
        'experiment': 'NAME', 'topology': 'TOPOLOGY', 'load': 'LOAD', 'k': 'K',
        'heur': 'HEUR',
    }
    heur_data = heur_data.rename(columns=_rename)
    bounds_data = bounds_data.rename(columns=_rename)
    # Keep only the columns needed
    _keep_cols = ['NAME', 'TOPOLOGY', 'LOAD', 'K', 'service_blocking_probability_mean',
                  'service_blocking_probability_std', 'service_blocking_probability_iqr_lower',
                  'service_blocking_probability_iqr_upper']
    heur_data = heur_data[[c for c in _keep_cols if c in heur_data.columns]]
    bounds_data = bounds_data[[c for c in _keep_cols + ['HEUR'] if c in bounds_data.columns]]

    def get_n_slots(name):
        return 40 if name == 'PtrNet-RSA-40' else 80 if name in ['PtrNet-RSA-80', 'MaskRSA'] else 100
    def get_topology(name):
        return 'JPN48' if 'JPN48' in name.upper() else 'COST239' if 'COST239' in name.upper() else 'USNET' if 'USNET' in name.upper() else 'NSFNET'
    def get_publication(name):
        if name in ['DeepRMSA', 'Reward-RMSA', 'GCN-RMSA']:
            return 'DeepRMSA~Reward-RMSA~GCN-RMSA'
        if 'PtrNet-RSA' in name:
            return 'PtrNet-RSA'
        return name
    # Add in the number of slots
    heur_data['N_slots'] = heur_data['NAME'].apply(get_n_slots)
    bounds_data['N_slots'] = bounds_data['NAME'].apply(get_n_slots)
    # Add in the topology
    heur_data['topology'] = heur_data['TOPOLOGY'].apply(get_topology)
    bounds_data['topology'] = bounds_data['TOPOLOGY'].apply(get_topology)
    # Add in the publication
    heur_data['publication'] = heur_data['NAME'].apply(get_publication)
    bounds_data['publication'] = bounds_data['NAME'].apply(get_publication)
    # Rename columns to match publication,topology,N_slots,load,mean,stddev
    heur_data = heur_data.rename(columns={'LOAD': 'load', 'K': 'k',
                            'service_blocking_probability_mean': 'mean',
                            'service_blocking_probability_std': 'stddev',
                            'service_blocking_probability_iqr_lower': 'iqr_lower',
                            'service_blocking_probability_iqr_upper': 'iqr_upper'})
    bounds_data = bounds_data.rename(columns={'LOAD': 'load', 'K': 'k',
                            'service_blocking_probability_mean': 'mean',
                            'service_blocking_probability_std': 'stddev',
                            'service_blocking_probability_iqr_lower': 'iqr_lower',
                            'service_blocking_probability_iqr_upper': 'iqr_upper'})

    # Sort by load before computing bands and plotting
    heur_data = heur_data.sort_values('load').reset_index(drop=True)
    bounds_data = bounds_data.sort_values('load').reset_index(drop=True)

    # Scale to percentages and compute mean ± SEM shaded band
    # n = number of independent samples (NUM_ENVS for heuristic, num_trials for bounds)
    def compute_bands(df, n):
        df['mean'] *= 100
        df['stddev'] *= 100
        sem = df['stddev'] / np.sqrt(n)
        df['band_lower'] = (df['mean'] - sem).clip(lower=0)
        df['band_upper'] = df['mean'] + sem
        return df
    heur_data = compute_bands(heur_data, n=2000)      # NUM_ENVS=2000
    bounds_data = compute_bands(bounds_data, n=10)     # num_trials=10

    # Define the publications and topologies
    publications = ['DeepRMSA~Reward-RMSA~GCN-RMSA', 'MaskRSA', 'PtrNet-RSA-40', 'PtrNet-RSA-80']
    topologies = ['NSFNET', 'COST239', 'USNET', 'JPN48']

    # purple
    heur_col = '#9467bd'
    # black
    bounds_col = '#000000'

    # Define grid
    grid = [
        ['NSFNET', 'COST239', 'USNET'],
        ['NSFNET', 'JPN48'],
        ['NSFNET', 'COST239', 'USNET'],
        ['NSFNET', 'COST239', 'USNET'],
    ]

    # Create a new figure
    fig = plt.figure(figsize=(30, 18))

    # Calculate the maximum number of rows
    max_rows = max(len(column) for column in grid)

    # To store all lines and labels for the legend
    all_lines = []
    all_labels = []

    # Loop through each subplot
    for col, publication in enumerate(publications):
        #pub = 'PtrNet-RSA' if 'PtrNet-RSA' in publication else publication
        n_slots = get_n_slots(publication)

        col_def = grid[col]

        for n, topology in enumerate(col_def):
            # Calculate the position of the subplot
            ax = fig.add_subplot(max_rows, len(publications), col + 1 + n * len(publications))

            ax.set_yscale('log')
            # show grid for y-axis
            ax.yaxis.grid(True)
            ax.set_xticks(np.arange(0, 1000, 25))
            ax.set_ylim(0.01, 1)

            # Plot the case and check if data was plotted
            data_plotted, lines, labels = plot_case(ax, publication, topology, n_slots, heur_data, bounds_data)

            # If no data was plotted, remove the subplot
            if not data_plotted:
                fig.delaxes(ax)
            else:
                all_lines.extend(lines)
                all_labels.extend(labels)

    # Remove duplicate labels while preserving order
    unique_labels = []
    unique_lines = []
    for label, line in zip(all_labels, all_lines):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_lines.append(line)

    # Add common x and y labels with adjusted position
    # make font boldface
    fig.text(0.51, 0.09, 'Traffic Load (Erlang)', ha='center', va='center', fontsize=36)
    fig.text(0.025, 0.48, 'Service Blocking Probability (%)', ha='center', va='center', rotation='vertical', fontsize=36)

    # Add a single, stacked legend to the figure
    #legend = fig.legend(unique_lines, unique_labels, loc='lower left', bbox_to_anchor=(0.048, 0.07), ncol=1)  # Set ncol=1 for a stacked legend
    legend = fig.legend(unique_lines, unique_labels, loc='lower center', ncol=5)# bbox_to_anchor=(0.5, 0.95), ncol=4)  # Set ncol=1 for a stacked legend

    # Increase the line thickness in the legend
    increase_legend_line_thickness(legend, line_width=6, marker_size=15)  # Adjust line_width as needed

    # Adjust layout and display the plot
    #plt.tight_layout(rect=[0.03, 0.04, 1, 0.99])  # Adjust the rect parameter to make room for labels and legend
    plt.tight_layout(rect=[0.03, 0.105, 1, 1])
    plt.savefig(PLOTS_DIR / 'bounds_comparison.png')

    # ---- New bounds comparison plot (results/bounds/) ----
    new_data_dir = Path(__file__).resolve().parents[1] / 'results' / 'bounds'
    new_reconfig_data = load_bounds_file(new_data_dir / 'experiment_results_reconfigurable_bounds')
    new_cutset_data = load_bounds_file(new_data_dir / 'experiment_results_cutsets_bounds')
    new_heur_data = load_bounds_file(new_data_dir / 'experiment_results_eval_bounds')

    _bounds_rename = {
        'experiment': 'NAME', 'topology': 'TOPOLOGY', 'load': 'LOAD', 'k': 'K',
        'heur': 'HEUR',
    }
    new_reconfig_data = new_reconfig_data.rename(columns=_bounds_rename)
    new_cutset_data = new_cutset_data.rename(columns=_bounds_rename)
    new_heur_data = new_heur_data.rename(columns=_bounds_rename)

    # Map individual experiment names to grouped publication names
    def get_publication_new(name):
        if name in ['DeepRMSA', 'Reward-RMSA', 'GCN-RMSA']:
            return 'DeepRMSA~Reward-RMSA~GCN-RMSA'
        if 'PtrNet-RSA' in name:
            return 'PtrNet-RSA'
        return name

    for df in [new_heur_data, new_reconfig_data, new_cutset_data]:
        df['N_slots'] = df['NAME'].apply(get_n_slots)
        df['topology'] = df['TOPOLOGY'].apply(get_topology)
        df['publication'] = df['NAME'].apply(get_publication_new)
        df.rename(columns={'LOAD': 'load', 'K': 'k',
                           'service_blocking_probability_mean': 'mean',
                           'service_blocking_probability_std': 'stddev',
                           'service_blocking_probability_iqr_lower': 'iqr_lower',
                           'service_blocking_probability_iqr_upper': 'iqr_upper'}, inplace=True)

    # Load transformer model evaluation data
    new_transformer_data = load_bounds_file(new_data_dir / 'experiment_results_transformer_eval_bounds')
    new_transformer_data = new_transformer_data.rename(columns=_bounds_rename)

    # Map (topology_name, link_resources) -> NAME to match existing publication scheme
    def _map_transformer_name(row):
        topo = row['TOPOLOGY']
        lr = row['link_resources']
        if 'directed' in topo and 'undirected' not in topo:
            if 'usnet' in topo:
                return 'GCN-RMSA'
            return 'DeepRMSA'
        if 'jpn48' in topo:
            return 'MaskRSA'
        if lr == 40:
            return 'PtrNet-RSA-40'
        if lr == 80:
            if 'nsfnet' in topo:
                return None  # handled by duplication below
            return 'PtrNet-RSA-80'
        return 'DeepRMSA'

    new_transformer_data['NAME'] = new_transformer_data.apply(_map_transformer_name, axis=1)
    # Split nsfnet_deeprmsa_undirected/80 by load range:
    # low loads (<=180) → MaskRSA, high loads (>180) → PtrNet-RSA-80
    nsfnet_80 = new_transformer_data[
        (new_transformer_data['TOPOLOGY'].str.contains('nsfnet')) &
        (new_transformer_data['link_resources'] == 80) &
        (new_transformer_data['NAME'].isna())
    ]
    mask_rows = nsfnet_80[nsfnet_80['LOAD'] <= 180].copy()
    mask_rows['NAME'] = 'MaskRSA'
    ptr80_rows = nsfnet_80[nsfnet_80['LOAD'] > 180].copy()
    ptr80_rows['NAME'] = 'PtrNet-RSA-80'
    new_transformer_data = pd.concat([
        new_transformer_data.dropna(subset=['NAME']),
        mask_rows, ptr80_rows
    ], ignore_index=True)
    new_transformer_data.drop(columns=['link_resources'], inplace=True)

    for df in [new_transformer_data]:
        df['N_slots'] = df['NAME'].apply(get_n_slots)
        df['topology'] = df['TOPOLOGY'].apply(get_topology)
        df['publication'] = df['NAME'].apply(get_publication_new)
        df.rename(columns={'LOAD': 'load', 'K': 'k',
                           'service_blocking_probability_mean': 'mean',
                           'service_blocking_probability_std': 'stddev',
                           'service_blocking_probability_iqr_lower': 'iqr_lower',
                           'service_blocking_probability_iqr_upper': 'iqr_upper'}, inplace=True)

    # Sort all dataframes by load before computing bands and plotting
    new_heur_data = new_heur_data.sort_values('load').reset_index(drop=True)
    new_reconfig_data = new_reconfig_data.sort_values('load').reset_index(drop=True)
    new_cutset_data = new_cutset_data.sort_values('load').reset_index(drop=True)
    new_transformer_data = new_transformer_data.sort_values('load').reset_index(drop=True)

    new_heur_data = compute_bands(new_heur_data, n=200)       # NUM_ENVS=200
    new_reconfig_data = compute_bands(new_reconfig_data, n=10)  # num_trials=10
    new_cutset_data = compute_bands(new_cutset_data, n=10)      # num_trials=10
    new_transformer_data = compute_bands(new_transformer_data, n=200)  # NUM_ENVS=200
    cutset_col = '#30A08E'
    transformer_col = '#d62728'  # red

    def plot_case_new(ax, pub, topology, n_slots, heur_df, reconfig_df, cutset_df,
                      transformer_df=None):
        pub_filter = 'PtrNet-RSA' if 'PtrNet-RSA' in pub else pub

        case_heur = heur_df[(heur_df['publication'] == pub_filter) &
                            (heur_df['topology'] == topology) &
                            (heur_df['N_slots'] == n_slots)]
        case_reconfig = reconfig_df[(reconfig_df['publication'] == pub_filter) &
                                     (reconfig_df['topology'] == topology) &
                                     (reconfig_df['N_slots'] == n_slots)]
        case_cutset = cutset_df[(cutset_df['publication'] == pub_filter) &
                                 (cutset_df['topology'] == topology) &
                                 (cutset_df['N_slots'] == n_slots)]
        case_transformer = pd.DataFrame()
        if transformer_df is not None:
            case_transformer = transformer_df[
                (transformer_df['publication'] == pub_filter) &
                (transformer_df['topology'] == topology) &
                (transformer_df['N_slots'] == n_slots)]

        lines = []
        labels = []

        if not case_heur.empty:
            line = ax.plot(case_heur['load'], case_heur['mean'],
                          marker='o', markerfacecolor=heur_col, linestyle='-', color=heur_col)
            ax.fill_between(case_heur['load'], case_heur['band_lower'],
                           case_heur['band_upper'], alpha=0.2, color=heur_col)
            lines.append(line[0])
            labels.append('Best heuristic')

        if not case_transformer.empty:
            line = ax.plot(case_transformer['load'], case_transformer['mean'],
                          marker='D', markerfacecolor=transformer_col, linestyle='-',
                          color=transformer_col)
            ax.fill_between(case_transformer['load'], case_transformer['band_lower'],
                           case_transformer['band_upper'], alpha=0.2, color=transformer_col)
            lines.append(line[0])
            labels.append('Transformer')

        if not case_reconfig.empty:
            line = ax.plot(case_reconfig['load'], case_reconfig['mean'],
                          marker='s', markerfacecolor=bounds_col, linestyle='-', color=bounds_col)
            ax.fill_between(case_reconfig['load'], case_reconfig['band_lower'],
                           case_reconfig['band_upper'], alpha=0.2, color=bounds_col)
            lines.append(line[0])
            labels.append('Defragmentation bound')

        if not case_cutset.empty:
            line = ax.plot(case_cutset['load'], case_cutset['mean'],
                          marker='^', markerfacecolor=cutset_col, linestyle='-', color=cutset_col)
            ax.fill_between(case_cutset['load'], case_cutset['band_lower'],
                           case_cutset['band_upper'], alpha=0.2, color=cutset_col)
            lines.append(line[0])
            labels.append('Cut-set bound')

        # Dynamically adjust axis limits so at least one point per series is visible.
        # Start from the default limits and only expand.
        y_min, y_max = 0.01, 1
        x_min, x_max = ax.get_xlim()  # auto-scaled from plotted data
        for case in [case_heur, case_reconfig, case_cutset, case_transformer]:
            if case.empty:
                continue
            visible = case[case['mean'] > 0]
            if visible.empty:
                continue
            # Pick the point with the largest mean (most likely already in range)
            best_idx = visible['mean'].idxmax()
            best_y = visible.loc[best_idx, 'mean']
            best_x = visible.loc[best_idx, 'load']
            if best_y < y_min:
                y_min = best_y * 0.5  # margin below on log scale
            if best_y > y_max:
                y_max = best_y * 2.0  # margin above on log scale
            if best_x > x_max:
                x_max = best_x + 25
            if best_x < x_min:
                x_min = best_x - 25
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)

        pub_display = 'Deep/Reward/GCN-RMSA' if pub == 'DeepRMSA~Reward-RMSA~GCN-RMSA' else pub
        title = f'{pub_display}\n\n{topology}' if topology == 'NSFNET' else topology
        ax.set_title(title, fontsize=32)

        has_data = not (case_heur.empty and case_reconfig.empty and
                        case_cutset.empty and case_transformer.empty)
        return has_data, lines, labels

    # Create new figure with same grid layout
    fig2 = plt.figure(figsize=(30, 18))
    all_lines2 = []
    all_labels2 = []

    for col, publication in enumerate(publications):
        n_slots = get_n_slots(publication)
        col_def = grid[col]

        for n, topology in enumerate(col_def):
            ax = fig2.add_subplot(max_rows, len(publications), col + 1 + n * len(publications))
            ax.set_yscale('log')
            ax.yaxis.grid(True)
            ax.set_xticks(np.arange(0, 1000, 25))

            data_plotted, lines, labels = plot_case_new(
                ax, publication, topology, n_slots,
                new_heur_data, new_reconfig_data, new_cutset_data,
                new_transformer_data
            )

            if not data_plotted:
                fig2.delaxes(ax)
            else:
                all_lines2.extend(lines)
                all_labels2.extend(labels)

    unique_labels2 = []
    unique_lines2 = []
    for label, line in zip(all_labels2, all_lines2):
        if label not in unique_labels2:
            unique_labels2.append(label)
            unique_lines2.append(line)

    fig2.text(0.51, 0.09, 'Traffic Load (Erlang)', ha='center', va='center', fontsize=36)
    fig2.text(0.025, 0.48, 'Service Blocking Probability (%)', ha='center', va='center', rotation='vertical', fontsize=36)
    legend2 = fig2.legend(unique_lines2, unique_labels2, loc='lower center', ncol=5)
    increase_legend_line_thickness(legend2, line_width=6, marker_size=15)
    plt.tight_layout(rect=[0.03, 0.105, 1, 1])
    plt.savefig(PLOTS_DIR / 'bounds_comparison_new.png')

    # ---- bounds_comparison_new_with_rl: same as bounds_comparison_new + RL series ----
    # Published RL results digitised from the original papers (already in %)
    csv_data_rl = """
publication,topology,N_slots,load,mean,stddev
DeepRMSA,NSFNET,100,250,4.00,0
DeepRMSA,COST239,100,600,5.75,0
Reward-RMSA,NSFNET,100,168,0.35,0
Reward-RMSA,NSFNET,100,182,0.70,0
Reward-RMSA,NSFNET,100,196,1.20,0
Reward-RMSA,NSFNET,100,210,1.80,0
GCN-RMSA,NSFNET,100,154,0.51,0
GCN-RMSA,NSFNET,100,168,0.66,0
GCN-RMSA,NSFNET,100,182,0.98,0
GCN-RMSA,NSFNET,100,196,1.50,0
GCN-RMSA,NSFNET,100,210,2.10,0
GCN-RMSA,COST239,100,368,0.71,0
GCN-RMSA,COST239,100,391,0.85,0
GCN-RMSA,COST239,100,414,1.25,0
GCN-RMSA,COST239,100,437,1.75,0
GCN-RMSA,COST239,100,460,2.10,0
GCN-RMSA,USNET,100,320,0.65,0
GCN-RMSA,USNET,100,340,0.83,0
GCN-RMSA,USNET,100,360,1.05,0
GCN-RMSA,USNET,100,380,1.60,0
GCN-RMSA,USNET,100,400,2.20,0
MaskRSA,NSFNET,80,80,0.01,0
MaskRSA,NSFNET,80,90,0.05,0
MaskRSA,NSFNET,80,100,0.20,0
MaskRSA,NSFNET,80,110,0.50,0
MaskRSA,NSFNET,80,120,0.79,0
MaskRSA,NSFNET,80,130,1.80,0
MaskRSA,NSFNET,80,140,3.00,0
MaskRSA,NSFNET,80,150,4.30,0
MaskRSA,NSFNET,80,160,5.30,0
MaskRSA,JPN48,80,120,0.05,0
MaskRSA,JPN48,80,130,0.20,0
MaskRSA,JPN48,80,140,0.55,0
MaskRSA,JPN48,80,150,0.90,0
MaskRSA,JPN48,80,160,1.60,0
PtrNet-RSA,NSFNET,40,180,0.01,0
PtrNet-RSA,NSFNET,40,190,0.03,0
PtrNet-RSA,NSFNET,40,200,0.08,0
PtrNet-RSA,NSFNET,40,210,0.19,0
PtrNet-RSA,NSFNET,40,220,0.23,0
PtrNet-RSA,NSFNET,40,230,0.75,0
PtrNet-RSA,NSFNET,40,240,1.30,0
PtrNet-RSA,COST239,40,340,0.01,0
PtrNet-RSA,COST239,40,360,0.04,0
PtrNet-RSA,COST239,40,380,0.12,0
PtrNet-RSA,COST239,40,400,0.24,0
PtrNet-RSA,COST239,40,420,0.39,0
PtrNet-RSA,USNET,40,210,0.01,0
PtrNet-RSA,USNET,40,220,0.08,0
PtrNet-RSA,USNET,40,230,0.22,0
PtrNet-RSA,USNET,40,240,0.38,0
PtrNet-RSA,USNET,40,250,0.68,0
PtrNet-RSA,USNET,40,260,1.10,0
PtrNet-RSA,USNET,40,270,1.80,0
PtrNet-RSA,USNET,40,280,2.20,0
PtrNet-RSA,NSFNET,80,200,0.01,0
PtrNet-RSA,NSFNET,80,210,0.03,0
PtrNet-RSA,NSFNET,80,220,0.11,0
PtrNet-RSA,NSFNET,80,230,0.16,0
PtrNet-RSA,NSFNET,80,240,0.50,0
PtrNet-RSA,COST239,80,420,0.01,0
PtrNet-RSA,COST239,80,440,0.16,0
PtrNet-RSA,COST239,80,460,0.65,0
PtrNet-RSA,USNET,80,260,0.03,0
PtrNet-RSA,USNET,80,270,0.09,0
PtrNet-RSA,USNET,80,280,0.15,0
PtrNet-RSA,USNET,80,290,0.29,0
PtrNet-RSA,USNET,80,300,0.52,0
PtrNet-RSA,USNET,80,310,0.66,0
PtrNet-RSA,USNET,80,320,1.00,0
PtrNet-RSA,USNET,80,330,1.50,0
"""
    df_rl = pd.read_csv(StringIO(csv_data_rl))
    # Data is already in %; add band columns for consistency
    df_rl['band_lower'] = df_rl['mean']
    df_rl['band_upper'] = df_rl['mean']

    rl_col = '#d62728'  # red

    # Map RL publication names to the grouped scheme used in bounds_comparison_new
    # Map RL publication names to pub_filter-style keys used in filtering
    _rl_pub_to_group = {
        'DeepRMSA': 'DeepRMSA~Reward-RMSA~GCN-RMSA',
        'Reward-RMSA': 'DeepRMSA~Reward-RMSA~GCN-RMSA',
        'GCN-RMSA': 'DeepRMSA~Reward-RMSA~GCN-RMSA',
        'MaskRSA': 'MaskRSA',
        'PtrNet-RSA': 'PtrNet-RSA',
    }
    df_rl['group'] = df_rl['publication'].map(_rl_pub_to_group)

    def plot_case_new_with_rl(ax, pub, topology, n_slots, heur_df, reconfig_df,
                              cutset_df, transformer_df, rl_df,
                              cutset_max_load=None):
        """Like plot_case_new but also plots the RL series."""
        pub_filter = 'PtrNet-RSA' if 'PtrNet-RSA' in pub else pub

        case_heur = heur_df[(heur_df['publication'] == pub_filter) &
                            (heur_df['topology'] == topology) &
                            (heur_df['N_slots'] == n_slots)]
        case_reconfig = reconfig_df[(reconfig_df['publication'] == pub_filter) &
                                     (reconfig_df['topology'] == topology) &
                                     (reconfig_df['N_slots'] == n_slots)]
        case_cutset = cutset_df[(cutset_df['publication'] == pub_filter) &
                                 (cutset_df['topology'] == topology) &
                                 (cutset_df['N_slots'] == n_slots)]
        if cutset_max_load is not None and not case_cutset.empty:
            case_cutset = case_cutset[case_cutset['load'] <= cutset_max_load]
        case_transformer = pd.DataFrame()
        if transformer_df is not None:
            case_transformer = transformer_df[
                (transformer_df['publication'] == pub_filter) &
                (transformer_df['topology'] == topology) &
                (transformer_df['N_slots'] == n_slots)]

        # RL data: match using pub_filter (e.g. 'PtrNet-RSA' for both -40/-80)
        case_rl = rl_df[(rl_df['group'] == pub_filter) &
                        (rl_df['topology'] == topology) &
                        (rl_df['N_slots'] == n_slots)]

        lines = []
        labels = []

        # Plot RL series (one line per individual publication in group, single legend entry)
        if not case_rl.empty:
            rl_pubs = case_rl['publication'].unique()
            rl_markers = {'DeepRMSA': 'x', 'Reward-RMSA': '+', 'GCN-RMSA': '1'}
            first_rl = True
            for rl_pub in sorted(rl_pubs):
                sub = case_rl[case_rl['publication'] == rl_pub]
                marker = rl_markers.get(rl_pub, 'x')
                line = ax.plot(sub['load'], sub['mean'],
                               marker=marker, linestyle='-', color=rl_col,
                               markersize=12, linewidth=2)
                if first_rl:
                    lines.append(line[0])
                    labels.append('RL')
                    first_rl = False

        if not case_heur.empty:
            line = ax.plot(case_heur['load'], case_heur['mean'],
                           marker='o', markerfacecolor=heur_col, linestyle='-', color=heur_col)
            ax.fill_between(case_heur['load'], case_heur['band_lower'],
                            case_heur['band_upper'], alpha=0.2, color=heur_col)
            lines.append(line[0])
            labels.append('Best heuristic')

        if not case_transformer.empty:
            line = ax.plot(case_transformer['load'], case_transformer['mean'],
                           marker='D', markerfacecolor=transformer_col, linestyle='-',
                           color=transformer_col)
            ax.fill_between(case_transformer['load'], case_transformer['band_lower'],
                            case_transformer['band_upper'], alpha=0.2, color=transformer_col)
            lines.append(line[0])
            labels.append('Transformer RL')

        if not case_reconfig.empty:
            line = ax.plot(case_reconfig['load'], case_reconfig['mean'],
                           marker='s', markerfacecolor=bounds_col, linestyle='-', color=bounds_col)
            ax.fill_between(case_reconfig['load'], case_reconfig['band_lower'],
                            case_reconfig['band_upper'], alpha=0.2, color=bounds_col)
            lines.append(line[0])
            labels.append('Defragmentation bound')

        if not case_cutset.empty:
            line = ax.plot(case_cutset['load'], case_cutset['mean'],
                           marker='^', markerfacecolor=cutset_col, linestyle='-', color=cutset_col)
            ax.fill_between(case_cutset['load'], case_cutset['band_lower'],
                            case_cutset['band_upper'], alpha=0.2, color=cutset_col)
            lines.append(line[0])
            labels.append('Cut-set bound')

        # Dynamically adjust axis limits
        y_min, y_max = 0.01, 1
        x_min, x_max = ax.get_xlim()
        for case in [case_heur, case_reconfig, case_cutset, case_transformer, case_rl]:
            if case.empty:
                continue
            visible = case[case['mean'] > 0]
            if visible.empty:
                continue
            best_idx = visible['mean'].idxmax()
            best_y = visible.loc[best_idx, 'mean']
            best_x = visible.loc[best_idx, 'load']
            if best_y < y_min:
                y_min = best_y * 0.5
            if best_y > y_max:
                y_max = best_y * 2.0
            if best_x > x_max:
                x_max = best_x + 25
            if best_x < x_min:
                x_min = best_x - 25
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)

        pub_display = 'Deep/Reward/GCN-RMSA' if pub == 'DeepRMSA~Reward-RMSA~GCN-RMSA' else pub
        title = f'{pub_display}\n\n{topology}' if topology == 'NSFNET' else topology
        ax.set_title(title, fontsize=32)

        has_data = not (case_heur.empty and case_reconfig.empty and
                        case_cutset.empty and case_transformer.empty and case_rl.empty)
        return has_data, lines, labels

    # Create figure with same grid layout
    fig3 = plt.figure(figsize=(30, 18))
    all_lines3 = []
    all_labels3 = []

    for col, publication in enumerate(publications):
        n_slots = get_n_slots(publication)
        col_def = grid[col]

        for n, topology in enumerate(col_def):
            ax = fig3.add_subplot(max_rows, len(publications), col + 1 + n * len(publications))
            ax.set_yscale('log')
            ax.yaxis.grid(True)

            # Per-subplot x-tick spacing: use 50 for wide-range subplots
            pub_display = 'Deep/Reward/GCN-RMSA' if publication == 'DeepRMSA~Reward-RMSA~GCN-RMSA' else publication
            wide_tick_cases = (
                (pub_display == 'Deep/Reward/GCN-RMSA' and topology in ('COST239', 'USNET')),
                (publication == 'PtrNet-RSA-80' and topology in ('COST239', 'USNET')),
            )
            if any(wide_tick_cases):
                ax.set_xticks(np.arange(0, 1000, 50))
            else:
                ax.set_xticks(np.arange(0, 1000, 25))

            # Per-subplot cutset max load and x-axis limit overrides
            cutset_max_load = None
            xlim_max = None
            if publication == 'MaskRSA' and topology == 'JPN48':
                cutset_max_load = 280
                xlim_max = 290

            data_plotted, lines, labels = plot_case_new_with_rl(
                ax, publication, topology, n_slots,
                new_heur_data, new_reconfig_data, new_cutset_data,
                new_transformer_data, df_rl,
                cutset_max_load=cutset_max_load
            )

            if xlim_max is not None:
                ax.set_xlim(right=xlim_max)

            if not data_plotted:
                fig3.delaxes(ax)
            else:
                all_lines3.extend(lines)
                all_labels3.extend(labels)

    unique_labels3 = []
    unique_lines3 = []
    for label, line in zip(all_labels3, all_lines3):
        if label not in unique_labels3:
            unique_labels3.append(label)
            unique_lines3.append(line)

    fig3.text(0.51, 0.09, 'Traffic Load (Erlang)', ha='center', va='center', fontsize=36)
    fig3.text(0.025, 0.48, 'Service Blocking Probability (%)', ha='center', va='center', rotation='vertical', fontsize=36)
    legend3 = fig3.legend(unique_lines3, unique_labels3, loc='lower center', ncol=5)
    increase_legend_line_thickness(legend3, line_width=6, marker_size=15)
    plt.tight_layout(rect=[0.03, 0.105, 1, 1])
    plt.savefig(PLOTS_DIR / 'bounds_comparison_new_with_rl.png')
