import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.patches as mpatches

# Data (unchanged)
fractions_data = {
    'Topology': ['NSFNET', 'COST239', 'USNET', 'JPN48'],
    'k5_available': [0.34, 0.35, 0.28, 0.57],
    'k5_sp_ff': [0.050, 0.083, 0.105, 0.399],
    'k5_sp_ff_hops': [0.106, 0.207, 0.174, 0.488],
    'k5_sp_ff_err': [0.001, 0.010, 0.010, 0.016],
    'k5_sp_ff_hops_err': [0.010, 0.014, 0.012, 0.016],
    'k20_available': [0.28, 0.35, 0.24, 0.55],
    'k20_sp_ff': [0.006, 0.023, 0.011, 0.263],
    'k20_sp_ff_hops': [0.003, 0.069, 0.031, 0.338],
    'k20_sp_ff_err': [0.003, 0.005, 0.003, 0.015],
    'k20_sp_ff_hops_err': [0.002, 0.009, 0.006, 0.015]
}

path_hops_data = {
    'Topology': ['NSFNET', 'COST239', 'USNET', 'JPN48'],
    '5-SP-FF_length': [2029, 1700, 923, 1087],
    '5-SP-FF_hops': [2.44, 1.82, 3.09, 6.20],
    '5-SP-FF_length_err': [1047, 746, 451, 710],
    '5-SP-FF_hops_err': [1.13, 0.84, 1.48, 3.76],
    '5-SP-FF_hops_length': [2357, 2083, 1032, 1296],
    '5-SP-FF_hops_hops': [2.21, 1.64, 3.03, 5.24],
    '5-SP-FF_hops_length_err': [1351, 1109, 534, 857],
    '5-SP-FF_hops_hops_err': [0.86, 0.60, 1.37, 2.94],
    '20-SP-FF_length': [2133, 1752, 948, 1094],
    '20-SP-FF_hops': [2.57, 1.88, 3.18, 6.23],
    '20-SP-FF_length_err': [1149, 777, 456, 709],
    '20-SP-FF_hops_err': [1.27, 0.87, 1.50, 3.75],
    '20-SP-FF_hops_length': [2379, 2122, 1040, 1305],
    '20-SP-FF_hops_hops': [2.23, 1.70, 3.07, 5.31],
    '20-SP-FF_hops_length_err': [1377, 1155, 536, 858],
    '20-SP-FF_hops_hops_err': [0.89, 0.67, 1.39, 2.96]
}

# Set up Helvetica font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams.update({'font.size': 28})
# Set axis label size
plt.rcParams.update({'axes.labelsize': 32})
# Set axis tick label size
plt.rcParams.update({'xtick.labelsize': 32})

# Define color scheme
topology_colors = {
    'NSFNET': '#1f77b4',
    'COST239': '#ff7f0e',
    'USNET': '#2ca02c',
    'JPN48': '#d62728'
}

# Update textures for different types
textures = {
    'available': '..',
    'sp_ff': 'xx',
    'sp_ff_hops': '//'
}


def plot_fractions_unique_paths():
    fig, ax = plt.subplots(figsize=(16, 10))

    x = np.arange(len(fractions_data['Topology']))
    width = 0.11

    labels = ['Available', 'Utilised SP-FF', r'Utilised SP-FF$_{hops}$']
    keys = ['available', 'sp_ff', 'sp_ff_hops']

    for i, topology in enumerate(fractions_data['Topology']):
        color = topology_colors[topology]
        for j, (label, key) in enumerate(zip(labels, keys)):
            k5_data = [fractions_data[f'k5_{key}'][i] * 100]
            k20_data = [fractions_data[f'k20_{key}'][i] * 100]
            if 'Available' not in label:
                k5_err = [fractions_data[f'k5_{key}_err'][i] * 100]
                k20_err = [fractions_data[f'k20_{key}_err'][i] * 100]
            else:
                k5_err = None
                k20_err = None

            # k=5 bars (solid color with texture)
            ax.bar([x[i] + j * width * 2], k5_data, width,
                   color=color, yerr=k5_err, capsize=5,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 1},
                   hatch=textures[key], edgecolor='white')

            # k=20 bars (outline with texture)
            ax.bar([x[i] + j * width * 2 + width], k20_data, width,
                   facecolor='none', edgecolor=color, yerr=k20_err, capsize=5,
                   error_kw={'elinewidth': 1, 'capthick': 1, 'alpha': 1},
                   hatch=textures[key], linewidth=1.5)

    ax.set_ylabel('Paths unique to path ordering (%)', labelpad=10)
    ax.set_xlabel('Topology', labelpad=10)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(fractions_data['Topology'])

    # Create custom legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['available'], label='Available'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['sp_ff'], label='Utilised: KSP-FF'),
        mpatches.Patch(facecolor='none', edgecolor='black', hatch=textures['sp_ff_hops'], label='Utilised: KSP-FF$_{hops}$'),
        mpatches.Patch(facecolor='black', label='K=5'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='K=20')
    ]

    # Add topology colors to legend
    #for topology, color in topology_colors.items():
    #    legend_elements.append(mpatches.Patch(facecolor=color, label=topology))

    ax.legend(handles=legend_elements, loc='upper left', ncol=2)

    plt.tight_layout()
    plt.show()


def plot_combined_dot_plot():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlabel('Path length (km)', labelpad=15)
    ax.set_ylabel('Path length (hops)')

    heuristic_markers = {'SP-FF': 'o', 'SP-FF_hops': 's'}
    k_styles = {'K=20': 'none', 'K=5': 'full'}

    for i, topology in enumerate(path_hops_data['Topology']):
        for k, style in k_styles.items():
            for heuristic, marker in heuristic_markers.items():
                x = path_hops_data[f'{"5" if k == "K=5" else "20"}-{heuristic}_length'][i]
                y = path_hops_data[f'{"5" if k == "K=5" else "20"}-{heuristic}_hops'][i]
                xerr = path_hops_data[f'{"5" if k == "K=5" else "20"}-{heuristic}_length_err'][i]
                yerr = path_hops_data[f'{"5" if k == "K=5" else "20"}-{heuristic}_hops_err'][i]

                markeredgewidth = 1 if k == "K=5" else 3
                alpha = 0.7 if k == "K=5" else 1

                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker,
                            color=topology_colors[topology], capsize=5, markersize=24,
                            label=f'{topology}, {k}, {heuristic}',
                            markerfacecolor=topology_colors[topology] if style == 'full' else 'white',
                            markeredgecolor=topology_colors[topology], markeredgewidth=markeredgewidth,
                            linestyle='--', elinewidth=0.25, capthick=2, alpha=alpha)

    # Customize legend
    topology_legend = [plt.Line2D([0], [0], color=color, label=topology, marker='', markersize=15, linestyle='-', linewidth=10)
                       for topology, color in topology_colors.items()]
    heuristic_legend = [
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF'], color='black', label='5-SP-FF', markersize=15,
                   linestyle=''),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF'], color='black', label='20-SP-FF', markersize=15,
                   linestyle='', markerfacecolor='none'),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF_hops'], color='black', label=r'5-SP-FF$_{hops}$',
                   markersize=15, linestyle=''),
        plt.Line2D([0], [0], marker=heuristic_markers['SP-FF_hops'], color='black', label=r'20-SP-FF$_{hops}$',
                   markersize=15, linestyle='', markerfacecolor='none')]

    ax.legend(handles=topology_legend + heuristic_legend, loc='upper right', markerscale=1.5)

    plt.tight_layout()
    plt.show()


def main():
    plot_fractions_unique_paths()
    plot_combined_dot_plot()


if __name__ == '__main__':
    main()