import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Set up Helvetica font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'axes.labelsize': 32})
plt.rcParams.update({'xtick.labelsize': 32})

topology_colors = {
   'NSFNET': '#264653',  # Dark teal
   'COST239': '#2A9D8F', # Seafoam
   'USNET': '#E9C46A',   # Gold
   'JPN48': '#F4A261'    # Coral
}

# Update textures for different types
textures = {
    'available': '..',
    'sp_ff': 'xx',
    'sp_ff_hops': '//'
}


def process_csv_data(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Extract topology names and clean them
    topology_mapping = {
        'nsfnet_deeprmsa_directed': 'NSFNET',
        'cost239_deeprmsa_directed': 'COST239',
        'usnet_gcnrnn_directed': 'USNET',
        'jpn48_directed': 'JPN48'
    }

    # Create processed dataframes for each K value
    k5_data = df[df['K'] == 5].copy()
    k50_data = df[df['K'] == 50].copy()  # Using K=50 data as substitute for K=50

    # Map topology names
    k5_data['TOPOLOGY'] = k5_data['TOPOLOGY'].map(topology_mapping)
    k50_data['TOPOLOGY'] = k50_data['TOPOLOGY'].map(topology_mapping)

    # Process data for fractions
    fractions_data = {
        'Topology': list(topology_mapping.values()),
        'k5_available': [],
        'k50_available': [],
        'k5_sp_ff': [],
        'k50_sp_ff': [],
        'k5_sp_ff_iqr_lower': [],
        'k5_sp_ff_iqr_upper': [],
        'k50_sp_ff_iqr_lower': [],
        'k50_sp_ff_iqr_upper': [],
        'k5_sp_ff_err': [],
        'k50_sp_ff_err': [],
        'k5_sp_ff_hops': [],
        'k50_sp_ff_hops': [],
        'k5_sp_ff_hops_err': [],
        'k50_sp_ff_hops_err': [],
        'k5_sp_ff_hops_iqr_lower': [],
        'k5_sp_ff_hops_iqr_upper': [],
        'k50_sp_ff_hops_iqr_lower': [],
        'k50_sp_ff_hops_iqr_upper': []
    }

    path_hops_data = {
        'Topology': list(topology_mapping.values()),
        '5-SP-FF_length': [],
        '5-SP-FF_hops': [],
        '5-SP-FF_length_err': [],
        '5-SP-FF_hops_err': [],
        '5-SP-FF_length_iqr_lower': [],
        '5-SP-FF_length_iqr_upper': [],
        '5-SP-FF_hops_iqr_lower': [],
        '5-SP-FF_hops_iqr_upper': [],
        '50-SP-FF_length': [],
        '50-SP-FF_hops': [],
        '50-SP-FF_length_err': [],
        '50-SP-FF_hops_err': [],
        '50-SP-FF_length_iqr_lower': [],
        '50-SP-FF_length_iqr_upper': [],
        '50-SP-FF_hops_iqr_lower': [],
        '50-SP-FF_hops_iqr_upper': [],
        '5-SP-FF_hops_length': [],
        '5-SP-FF_hops_hops': [],
        '5-SP-FF_hops_length_err': [],
        '5-SP-FF_hops_hops_err': [],
        '5-SP-FF_hops_length_iqr_lower': [],
        '5-SP-FF_hops_length_iqr_upper': [],
        '5-SP-FF_hops_hops_iqr_lower': [],
        '5-SP-FF_hops_hops_iqr_upper': [],
        '50-SP-FF_hops_length': [],
        '50-SP-FF_hops_hops': [],
        '50-SP-FF_hops_length_err': [],
        '50-SP-FF_hops_hops_err': [],
        '50-SP-FF_hops_length_iqr_lower': [],
        '50-SP-FF_hops_length_iqr_upper': [],
        '50-SP-FF_hops_hops_iqr_lower': [],
        '50-SP-FF_hops_hops_iqr_upper': []
    }

    for topology in topology_mapping.values():
        # Process k5 data
        k5_topology_data = k5_data[k5_data['TOPOLOGY'] == topology]
        k50_topology_data = k50_data[k50_data['TOPOLOGY'] == topology]

        # Fractions data
        fractions_data['k5_available'].append(k5_topology_data['unique_paths_fraction'].iloc[0])
        fractions_data['k50_available'].append(k50_topology_data['unique_paths_fraction'].iloc[0])

        fractions_data['k5_sp_ff'].append(k5_topology_data['successful_unique_paths_mean'].iloc[0])
        fractions_data['k50_sp_ff'].append(k50_topology_data['successful_unique_paths_mean'].iloc[0])

        fractions_data['k5_sp_ff_err'].append(k5_topology_data['successful_unique_paths_std'].iloc[0])
        fractions_data['k50_sp_ff_err'].append(k50_topology_data['successful_unique_paths_std'].iloc[0])

        fractions_data['k5_sp_ff_iqr_lower'].append(k5_topology_data['successful_unique_paths_iqr_lower'].iloc[0])
        fractions_data['k5_sp_ff_iqr_upper'].append(k5_topology_data['successful_unique_paths_iqr_upper'].iloc[0])
        fractions_data['k50_sp_ff_iqr_lower'].append(k50_topology_data['successful_unique_paths_iqr_lower'].iloc[0])
        fractions_data['k50_sp_ff_iqr_upper'].append(k50_topology_data['successful_unique_paths_iqr_upper'].iloc[0])

        fractions_data['k5_sp_ff_hops'].append(k5_topology_data['successful_unique_paths_mean'].iloc[1])
        fractions_data['k50_sp_ff_hops'].append(k50_topology_data['successful_unique_paths_mean'].iloc[1])

        fractions_data['k5_sp_ff_hops_err'].append(k5_topology_data['successful_unique_paths_std'].iloc[1])
        fractions_data['k50_sp_ff_hops_err'].append(k50_topology_data['successful_unique_paths_std'].iloc[1])

        fractions_data['k5_sp_ff_hops_iqr_lower'].append(k5_topology_data['successful_unique_paths_iqr_lower'].iloc[1])
        fractions_data['k5_sp_ff_hops_iqr_upper'].append(k5_topology_data['successful_unique_paths_iqr_upper'].iloc[1])
        fractions_data['k50_sp_ff_hops_iqr_lower'].append(k50_topology_data['successful_unique_paths_iqr_lower'].iloc[1])
        fractions_data['k50_sp_ff_hops_iqr_upper'].append(k50_topology_data['successful_unique_paths_iqr_upper'].iloc[1])

        # Path and hops data
        path_hops_data['5-SP-FF_length'].append(k5_topology_data['avg_path_length_mean'].iloc[0])
        path_hops_data['5-SP-FF_hops'].append(k5_topology_data['avg_hops_mean'].iloc[0])
        path_hops_data['5-SP-FF_length_err'].append(k5_topology_data['avg_path_length_std'].iloc[0])
        path_hops_data['5-SP-FF_hops_err'].append(k5_topology_data['avg_hops_std'].iloc[0])
        path_hops_data['5-SP-FF_length_iqr_lower'].append(k5_topology_data['avg_path_length_iqr_lower'].iloc[0])
        path_hops_data['5-SP-FF_length_iqr_upper'].append(k5_topology_data['avg_path_length_iqr_upper'].iloc[0])
        path_hops_data['5-SP-FF_hops_iqr_lower'].append(k5_topology_data['avg_hops_iqr_lower'].iloc[0])
        path_hops_data['5-SP-FF_hops_iqr_upper'].append(k5_topology_data['avg_hops_iqr_upper'].iloc[0])

        path_hops_data['50-SP-FF_length'].append(k50_topology_data['avg_path_length_mean'].iloc[0])
        path_hops_data['50-SP-FF_hops'].append(k50_topology_data['avg_hops_mean'].iloc[0])
        path_hops_data['50-SP-FF_length_err'].append(k50_topology_data['avg_path_length_std'].iloc[0])
        path_hops_data['50-SP-FF_hops_err'].append(k50_topology_data['avg_hops_std'].iloc[0])
        path_hops_data['50-SP-FF_length_iqr_lower'].append(k50_topology_data['avg_path_length_iqr_lower'].iloc[0])
        path_hops_data['50-SP-FF_length_iqr_upper'].append(k50_topology_data['avg_path_length_iqr_upper'].iloc[0])
        path_hops_data['50-SP-FF_hops_iqr_lower'].append(k50_topology_data['avg_hops_iqr_lower'].iloc[0])
        path_hops_data['50-SP-FF_hops_iqr_upper'].append(k50_topology_data['avg_hops_iqr_upper'].iloc[0])

        # Hops variant data
        path_hops_data['5-SP-FF_hops_length'].append(k5_topology_data['avg_path_length_mean'].iloc[1])
        path_hops_data['5-SP-FF_hops_hops'].append(k5_topology_data['avg_hops_mean'].iloc[1])
        path_hops_data['5-SP-FF_hops_length_err'].append(k5_topology_data['avg_path_length_std'].iloc[1])
        path_hops_data['5-SP-FF_hops_hops_err'].append(k5_topology_data['avg_hops_std'].iloc[1])
        path_hops_data['5-SP-FF_hops_length_iqr_lower'].append(k5_topology_data['avg_path_length_iqr_lower'].iloc[1])
        path_hops_data['5-SP-FF_hops_length_iqr_upper'].append(k5_topology_data['avg_path_length_iqr_upper'].iloc[1])
        path_hops_data['5-SP-FF_hops_hops_iqr_lower'].append(k5_topology_data['avg_hops_iqr_lower'].iloc[1])
        path_hops_data['5-SP-FF_hops_hops_iqr_upper'].append(k5_topology_data['avg_hops_iqr_upper'].iloc[1])

        path_hops_data['50-SP-FF_hops_length'].append(k50_topology_data['avg_path_length_mean'].iloc[1])
        path_hops_data['50-SP-FF_hops_hops'].append(k50_topology_data['avg_hops_mean'].iloc[1])
        path_hops_data['50-SP-FF_hops_length_err'].append(k50_topology_data['avg_path_length_std'].iloc[1])
        path_hops_data['50-SP-FF_hops_hops_err'].append(k50_topology_data['avg_hops_std'].iloc[1])
        path_hops_data['50-SP-FF_hops_length_iqr_lower'].append(k50_topology_data['avg_path_length_iqr_lower'].iloc[1])
        path_hops_data['50-SP-FF_hops_length_iqr_upper'].append(k50_topology_data['avg_path_length_iqr_upper'].iloc[1])
        path_hops_data['50-SP-FF_hops_hops_iqr_lower'].append(k50_topology_data['avg_hops_iqr_lower'].iloc[1])
        path_hops_data['50-SP-FF_hops_hops_iqr_upper'].append(k50_topology_data['avg_hops_iqr_upper'].iloc[1])

    return fractions_data, path_hops_data


def plot_fractions_unique_paths(fractions_data):
    fig, ax = plt.subplots(figsize=(16, 10))

    x = np.arange(len(fractions_data['Topology']))
    width = 0.11

    labels = ['Available', 'Utilised SP-FF', r'Utilised SP-FF$_{hops}$']
    keys = ['available', 'sp_ff', 'sp_ff_hops']

    for i, topology in enumerate(fractions_data['Topology']):
        color = topology_colors[topology]
        for j, (label, key) in enumerate(zip(labels, keys)):
            k5_data = [fractions_data[f'k5_{key}'][i] * 100]
            k50_data = [fractions_data[f'k50_{key}'][i] * 100]
            if 'Available' not in label:
                # k5_err = [fractions_data[f'k5_{key}_err'][i] * 100]
                # k50_err = [fractions_data[f'k50_{key}_err'][i] * 100]
                # Calculate asymmetric errors relative to the mean
                k5_err = [
                    [max(k5_data[0] - fractions_data[f'k5_{key}_iqr_lower'][i] * 100, 0)],  # distance from mean to lower bound
                    [max(fractions_data[f'k5_{key}_iqr_upper'][i] * 100 - k5_data[0], 0)]  # distance from mean to upper bound
                ]
                k50_err = [
                    [max(k50_data[0] - fractions_data[f'k50_{key}_iqr_lower'][i] * 100, 0)], # distance from mean to lower bound
                    [max(fractions_data[f'k50_{key}_iqr_upper'][i] * 100 - k50_data[0], 0)]  # distance from mean to upper bound
                ]
            else:
                k5_err = None
                k50_err = None

            # k=5 bars (solid color with texture)
            ax.bar([x[i] + j * width * 2], k5_data, width,
                   color=color, yerr=k5_err, capsize=5,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 1},
                   hatch=textures[key], edgecolor='white')

            # k=50 bars (outline with texture)
            ax.bar([x[i] + j * width * 2 + width], k50_data, width,
                   facecolor='none', edgecolor=color, yerr=k50_err, capsize=5,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 1},
                   hatch=textures[key], linewidth=1.5)

    ax.set_ylabel('Paths unique to path ordering (% of total)', labelpad=10)
    ax.set_xlabel('Topology', labelpad=10)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(fractions_data['Topology'])

    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['available'], label='Available'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['sp_ff'], label='Utilised: KSP-FF'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['sp_ff_hops'],
                       label='Utilised: KSP-FF$_{hops}$'),
        mpatches.Patch(facecolor='black', label='K=5'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='K=50')
    ]

    ax.legend(handles=legend_elements, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()


def plot_combined_dot_plot(path_hops_data):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlabel('Path length (km)', labelpad=15)
    ax.set_ylabel('Path length (hops)')

    heuristic_markers = {'SP-FF': 'o', 'SP-FF_hops': 's'}
    k_styles = {'K=50': 'none', 'K=5': 'full'}

    for i, topology in enumerate(path_hops_data['Topology']):
        for k, style in k_styles.items():
            for heuristic, marker in heuristic_markers.items():
                x = path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_length'][i]
                y = path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_hops'][i]

                # xerr = path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_length_err'][i]
                # yerr = path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_hops_err'][i]

                # Create asymmetrical error bars using IQR values
                xerr = [[x - path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_length_iqr_lower'][i]],
                        [path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_length_iqr_upper'][i] - x]]
                yerr = [[y - path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_hops_iqr_lower'][i]],
                        [path_hops_data[f'{"5" if k == "K=5" else "50"}-{heuristic}_hops_iqr_upper'][i] - y]]

                markeredgewidth = 1 if k == "K=5" else 3
                alpha = 0.7 if k == "K=5" else 1

                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker,
                            color=topology_colors[topology], capsize=5, markersize=24,
                            label=f'{topology}, {k}, {heuristic}',
                            markerfacecolor=topology_colors[topology] if style == 'full' else 'white',
                            markeredgecolor=topology_colors[topology], markeredgewidth=markeredgewidth,
                            linestyle='--', elinewidth=0.25, capthick=2, alpha=alpha)

    topology_legend = [
        plt.Line2D([0], [0], color=color, label=topology, marker='', markersize=15, linestyle='-', linewidth=10)
        for topology, color in topology_colors.items()]
    heuristic_legend = [
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF'], color='black', label='5-SP-FF', markersize=15,
                   linestyle=''),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF'], color='black', label='50-SP-FF', markersize=15,
                   linestyle='', markerfacecolor='none'),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF_hops'], color='black', label=r'5-SP-FF$_{hops}$',
                   markersize=15, linestyle=''),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF_hops'], color='black', label=r'50-SP-FF$_{hops}$',
                   markersize=15, linestyle='', markerfacecolor='none')]

    ax.legend(handles=topology_legend + heuristic_legend, loc='upper right', markerscale=1.5)
    plt.tight_layout()
    plt.show()


def main():
    # Read and process CSV data
    fractions_data, path_hops_data = process_csv_data('/Users/michaeldoherty/git/XLRON/data/JOCN2024/experiment_results_unique_paths.csv')

    # Create plots
    plot_fractions_unique_paths(fractions_data)
    plot_combined_dot_plot(path_hops_data)


if __name__ == '__main__':
    main()