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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import configure_style, increase_legend_line_thickness


def load_bounds_file(path):
    """Load a bounds results file, supporting both legacy CSV and JSONL formats.

    For JSONL files (produced by --DATA_OUTPUT_FILE), flattens the nested
    JSON structure into a flat DataFrame with columns matching the legacy
    CSV format (experiment, topology, load, k, plus metric columns).

    Returns a pandas DataFrame.
    """
    path = Path(path)

    # Try JSONL first if it exists, then fall back to CSV
    jsonl_path = path.with_suffix('.jsonl') if path.suffix != '.jsonl' else path
    csv_path = path.with_suffix('.csv') if path.suffix != '.csv' else path

    if jsonl_path.exists():
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row = {}
                config = obj.get('config', {})
                # Map config fields to legacy CSV column names
                row['experiment'] = config.get('PROJECT', '')
                row['topology'] = config.get('topology_name', '')
                row['load'] = config.get('load', 0)
                row['k'] = config.get('k', 0)
                row['heur'] = config.get('path_heuristic', '')
                # Flatten metrics: {metric_name: {stat: val}} -> metric_stat columns
                for metric_name, stats in obj.get('metrics', {}).items():
                    for stat_name, stat_val in stats.items():
                        row[f'{metric_name}_{stat_name}'] = stat_val
                records.append(row)
        return pd.DataFrame(records)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No bounds data file found at {jsonl_path} or {csv_path}"
        )

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

    data_dir = Path(__file__).resolve().parents[4] / 'experiment_data' / 'JOCN2024'
    heur_data = pd.read_csv(data_dir / 'experiment_results_eval_bounds.csv')
    bounds_data = pd.read_csv(data_dir / 'experiment_results_bounds.csv')

    # Filter to only have these columns: NAME,TOPOLOGY,LOAD,K,service_blocking_probability_mean/std/iqr_lower/iqr_upper
    heur_data = heur_data[['NAME', 'TOPOLOGY', 'LOAD', 'K', 'service_blocking_probability_mean', 'service_blocking_probability_std',
                'service_blocking_probability_iqr_lower', 'service_blocking_probability_iqr_upper']]
    # Rename bounds_data columns to match heur_data
    bounds_data = bounds_data.rename(columns={'experiment': 'NAME', 'topology': 'TOPOLOGY', 'load': 'LOAD', 'k': 'K',
                                              'heur': 'HEUR', 'blocking_prob_mean': 'service_blocking_probability_mean',
                                              'blocking_prob_std': 'service_blocking_probability_std',
                                              'blocking_prob_iqr_lower': 'service_blocking_probability_iqr_lower',
                                              'blocking_prob_iqr_upper': 'service_blocking_probability_iqr_upper'})

    def get_n_slots(name):
        return 40 if name == 'PtrNet-RSA-40' else 80 if name in ['PtrNet-RSA-80', 'MaskRSA'] else 100
    def get_topology(name):
        return 'JPN48' if 'JPN48' in name.upper() else 'COST239' if 'COST239' in name.upper() else 'USNET' if 'USNET' in name.upper() else 'NSFNET'
    def get_publication(name):
        return 'PtrNet-RSA' if 'PtrNet-RSA' in name else name
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

    # ---- New bounds comparison plot (experiment_data/bounds/) ----
    new_data_dir = Path(__file__).resolve().parents[4] / 'experiment_data' / 'bounds'
    new_reconfig_data = load_bounds_file(new_data_dir / 'experiment_results_reconfigurable_bounds.csv')
    new_cutset_data = load_bounds_file(new_data_dir / 'experiment_results_cutsets_bounds.csv')
    # Process heuristic eval data - CSV has 42 data fields for 41 headers
    # (extra empty field at position 5), so pandas uses field[0] as index.
    # The index has the true NAME, columns NAME/HEUR/TOPOLOGY/LOAD are shifted,
    # but metric columns (returns_mean onward) are correct due to offsets canceling.
    _raw_heur = pd.read_csv(new_data_dir / 'experiment_results_eval_bounds.csv')
    new_heur_data = pd.DataFrame({
        'NAME': _raw_heur.index,
        'TOPOLOGY': _raw_heur['HEUR'].values,
        'LOAD': _raw_heur['TOPOLOGY'].values,
        'K': _raw_heur['LOAD'].values,
        'service_blocking_probability_mean': _raw_heur['service_blocking_probability_mean'].values,
        'service_blocking_probability_std': _raw_heur['service_blocking_probability_std'].values,
        'service_blocking_probability_iqr_lower': _raw_heur['service_blocking_probability_iqr_lower'].values,
        'service_blocking_probability_iqr_upper': _raw_heur['service_blocking_probability_iqr_upper'].values,
    })

    # Rename columns to canonical names. Both old CSV columns (blocking_prob_*)
    # and new JSONL columns (service_blocking_probability_*) are handled;
    # pandas rename silently ignores columns that don't exist.
    _bounds_rename = {
        # common fields (CSV uses these; JSONL loader already produces them)
        'experiment': 'NAME', 'topology': 'TOPOLOGY', 'load': 'LOAD', 'k': 'K',
        'heur': 'HEUR',
        # old CSV metric columns -> canonical
        'blocking_prob_mean': 'service_blocking_probability_mean',
        'blocking_prob_std': 'service_blocking_probability_std',
        'blocking_prob_iqr_lower': 'service_blocking_probability_iqr_lower',
        'blocking_prob_iqr_upper': 'service_blocking_probability_iqr_upper',
    }
    new_reconfig_data = new_reconfig_data.rename(columns=_bounds_rename)
    new_cutset_data = new_cutset_data.rename(columns=_bounds_rename)

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
    new_transformer_data = load_bounds_file(new_data_dir / 'experiment_results_transformer_eval_bounds.jsonl')
    new_transformer_data = new_transformer_data.rename(columns=_bounds_rename)
    # The JSONL loader doesn't extract link_resources, so reload to get it
    _transformer_records = []
    with open(new_data_dir / 'experiment_results_transformer_eval_bounds.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            config = obj.get('config', {})
            _transformer_records.append(config.get('link_resources', 100))
    new_transformer_data['link_resources'] = _transformer_records

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

    new_heur_data = compute_bands(new_heur_data, n=2000)      # NUM_ENVS=2000
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
