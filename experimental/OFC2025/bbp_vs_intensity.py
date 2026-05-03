import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np


def process_data(data_str):
    # Convert string data to DataFrame
    df = pd.read_csv(io.StringIO(data_str))

    # Calculate traffic intensity
    df['traffic_intensity'] = (df['load'] * 0.75) / (44 * 11.5)

    return df


def plot_blocking_probability(df):
    plt.figure(figsize=(6, 4.5))
    plt.plot(df['traffic_intensity'], df['bitrate_blocking_probability_mean'],
             marker='o', linestyle='-', linewidth=2, markersize=8, label="0.5 dBm fixed")
    # Fill between IQR lower and upper
    plt.fill_between(df['traffic_intensity'], df['bitrate_blocking_probability_iqr_lower'],
                     df['bitrate_blocking_probability_iqr_upper'], alpha=0.3)

    plt.xlabel('Traffic Intensity', fontsize=12)
    plt.ylabel('Bitrate Blocking Probability', fontsize=12)
    #plt.title('Traffic Intensity vs Bitrate Blocking Probability', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)

    # Format axes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    formatter = plt.ScalarFormatter(useOffset=False, useMathText=False)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.tight_layout()

    plt.legend()

    return plt


def main():
    # Your data string goes here
    data = """load,bitrate_blocking_probability_mean,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper
67,0.07084,0.06732,0.07461
135,0.07092,0.06732,0.07464
202,0.07833,0.07348,0.08061
270,0.11229,0.10019,0.11319
337,0.16591,0.13761,0.18002
405,0.22111,0.17848,0.24107
472,0.26656,0.20715,0.30051
540,0.32148,0.23726,0.36877
607,0.35458,0.26333,0.42262"""

    # Process data and create plot
    df = process_data(data)
    plt = plot_blocking_probability(df)

    # Save the plot
    #plt.savefig('blocking_probability_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()