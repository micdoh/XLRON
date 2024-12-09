import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
import numpy as np
from scipy import interpolate


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
        ax.fill_between(case_heur['load'], case_heur['iqr_lower'],
                        case_heur['iqr_upper'], alpha=0.2, color=heur_col)
        lines.append(line[0])
        labels.append('Best heuristic')

    if not case_bounds.empty:
        line = ax.plot(case_bounds['load'], case_bounds['mean'], label='Bounds',
                       marker='o', markerfacecolor=bounds_col, linestyle='-', color=bounds_col)
        ax.fill_between(case_bounds['load'], case_bounds['iqr_lower'],
                        case_bounds['iqr_upper'], alpha=0.2, color=bounds_col)
        lines.append(line[0])
        labels.append('Defragmentation bound')

    # Add gap line at 0.1%
    add_gap_line(ax, pub, topology, n_slots, heur_data, bounds_data)

    pub = 'Deep/Reward/GCN-RMSA' if pub == 'DeepRMSA~Reward-RMSA~GCN-RMSA' else pub

    title = f'{pub}\n\n{topology}' if topology == 'NSFNET' else topology
    ax.set_title(title, fontsize=32)
    return True, lines, labels

def increase_legend_line_thickness(legend, line_width=3, marker_size=10):
    """
    Increase the line thickness and marker size in the legend without affecting the plot lines.

    :param legend: matplotlib.legend.Legend object
    :param line_width: desired line width for legend lines
    :param marker_size: desired marker size for legend markers
    """
    for n, handle in enumerate(legend.legend_handles):
        if isinstance(handle, mlines.Line2D):
            handle.set_linewidth(line_width)
            handle.set_markersize(25 if n < 2 else marker_size)
        elif isinstance(handle, mcollections.LineCollection):
            handle.set_linewidth(line_width)
        elif isinstance(handle, mcontainer.ErrorbarContainer):
            handle.lines[0].set_linewidth(line_width)
            if len(handle.lines) > 1:  # If there are caps on the error bars
                handle.lines[1].set_linewidth(line_width)
                handle.lines[2].set_linewidth(line_width)
            handle.lines[0].set_markersize(marker_size)


if __name__ == '__main__':
    # Set up Helvetica font and other plot parameters
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams.update({'font.size': 32})
    plt.rcParams.update({'axes.labelsize': 34})
    plt.rcParams.update({'xtick.labelsize': 24})
    plt.rcParams.update({'ytick.labelsize': 24})

    heur_data = pd.read_csv('/Users/michaeldoherty/git/XLRON/data/JOCN2024/experiment_results_eval_bounds.csv')
    bounds_data = pd.read_csv('/Users/michaeldoherty/git/XLRON/data/JOCN2024/experiment_results_bounds.csv')

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

    # Multiply mean, stddev, iqr_lower, iqr_upper by 100 to get percentages
    # Add 0.0001 to values to avoid zeros and ensure plotting
    def fix_iqrs(df):
        df['mean'] *= 100
        df['stddev'] *= 100
        df['iqr_lower'] *= 100
        df['iqr_upper'] *= 100
        # if IQR upper is zero, set it to df['mean']
        mask_mean = df[df['iqr_upper'] == 0]['mean']
        mask_stddev = df[df['iqr_upper'] == 0]['stddev']
        df.loc[mask_mean.index, 'iqr_upper'] = mask_mean
        df.loc[mask_mean.index, 'iqr_lower'] = mask_mean - mask_stddev
        return df
    heur_data = fix_iqrs(heur_data)
    bounds_data = fix_iqrs(bounds_data)

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
    plt.show()
