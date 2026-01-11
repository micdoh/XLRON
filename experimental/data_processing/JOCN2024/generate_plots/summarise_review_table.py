import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer

# CSV data (replace this with the actual content of your CSV file)
csv_data_hops = """
publication,topology,N_slots,load,mean,stddev,k
DeepRMSA,NSFNET,100,250,3.07,0.20,5
DeepRMSA,COST239,100,600,4.04,0.33,5
Reward-RMSA,NSFNET,100,168,0.13,0.06,5
Reward-RMSA,NSFNET,100,182,0.32,0.10,5
Reward-RMSA,NSFNET,100,196,0.66,0.22,5
Reward-RMSA,NSFNET,100,210,1.14,0.31,5
GCN-RMSA,NSFNET,100,154,0.06,0.05,5
GCN-RMSA,NSFNET,100,168,0.13,0.06,5
GCN-RMSA,NSFNET,100,182,0.32,0.10,5
GCN-RMSA,NSFNET,100,196,0.66,0.22,5
GCN-RMSA,NSFNET,100,210,1.14,0.31,5
GCN-RMSA,COST239,100,368,0.09,0.07,5
GCN-RMSA,COST239,100,391,0.14,0.09,5
GCN-RMSA,COST239,100,414,0.26,0.14,5
GCN-RMSA,COST239,100,437,0.49,0.26,5
GCN-RMSA,COST239,100,460,0.63,0.19,5
GCN-RMSA,USNET,100,320,0.23,0.13,5
GCN-RMSA,USNET,100,340,0.47,0.20,5
GCN-RMSA,USNET,100,360,0.74,0.10,5
GCN-RMSA,USNET,100,380,1.17,0.20,5
GCN-RMSA,USNET,100,400,1.62,0.21,5
MaskRSA,NSFNET,80,80,0.00,0.00,5
MaskRSA,NSFNET,80,90,0.01,0.01,5
MaskRSA,NSFNET,80,100,0.08,0.04,5
MaskRSA,NSFNET,80,110,0.25,0.08,5
MaskRSA,NSFNET,80,120,0.62,0.16,5
MaskRSA,NSFNET,80,130,1.38,0.21,5
MaskRSA,NSFNET,80,140,2.30,0.21,5
MaskRSA,NSFNET,80,150,3.37,0.31,5
MaskRSA,NSFNET,80,160,4.59,0.21,5
MaskRSA,JPN48,80,120,0.32,0.07,5
MaskRSA,JPN48,80,130,0.62,0.08,5
MaskRSA,JPN48,80,140,1.00,0.13,5
MaskRSA,JPN48,80,150,1.46,0.20,5
MaskRSA,JPN48,80,160,1.96,0.24,5
MaskRSA,JPN48,80,120,0.09,0.04,20
MaskRSA,JPN48,80,130,0.14,0.03,20
MaskRSA,JPN48,80,140,0.22,0.04,20
MaskRSA,JPN48,80,150,0.37,0.10,20
MaskRSA,JPN48,80,160,0.62,0.09,20
PtrNet-RSA,NSFNET,40,180,0.00,0.00,5
PtrNet-RSA,NSFNET,40,190,0.00,0.01,5
PtrNet-RSA,NSFNET,40,200,0.02,0.03,5
PtrNet-RSA,NSFNET,40,210,0.07,0.06,5
PtrNet-RSA,NSFNET,40,220,0.20,0.12,5
PtrNet-RSA,NSFNET,40,230,0.41,0.19,5
PtrNet-RSA,NSFNET,40,240,0.80,0.32,5
PtrNet-RSA,COST239,40,340,0.02,0.02,5
PtrNet-RSA,COST239,40,360,0.03,0.03,5
PtrNet-RSA,COST239,40,380,0.07,0.03,5
PtrNet-RSA,COST239,40,400,0.16,0.08,5
PtrNet-RSA,COST239,40,420,0.37,0.17,5
PtrNet-RSA,USNET,40,210,0.14,0.06,5
PtrNet-RSA,USNET,40,220,0.27,0.08,5
PtrNet-RSA,USNET,40,230,0.47,0.15,5
PtrNet-RSA,USNET,40,240,0.79,0.21,5
PtrNet-RSA,USNET,40,250,1.18,0.27,5
PtrNet-RSA,USNET,40,260,1.73,0.34,5
PtrNet-RSA,USNET,40,270,2.42,0.47,5
PtrNet-RSA,USNET,40,280,3.12,0.54,5
PtrNet-RSA,USNET,40,210,0.01,0.02,20
PtrNet-RSA,USNET,40,220,0.03,0.04,20
PtrNet-RSA,USNET,40,230,0.05,0.05,20
PtrNet-RSA,USNET,40,240,0.17,0.14,20
PtrNet-RSA,USNET,40,250,0.35,0.20,20
PtrNet-RSA,USNET,40,260,0.58,0.30,20
PtrNet-RSA,USNET,40,270,0.93,0.41,20
PtrNet-RSA,USNET,40,280,1.33,0.46,20
PtrNet-RSA,NSFNET,80,200,0.00,0.00,5
PtrNet-RSA,NSFNET,80,210,0.00,0.00,5
PtrNet-RSA,NSFNET,80,220,0.01,0.02,5
PtrNet-RSA,NSFNET,80,230,0.03,0.02,5
PtrNet-RSA,NSFNET,80,240,0.06,0.03,5
PtrNet-RSA,COST239,80,420,0.07,0.03,5
PtrNet-RSA,COST239,80,440,0.11,0.04,5
PtrNet-RSA,COST239,80,460,0.19,0.06,5
PtrNet-RSA,USNET,80,260,0.47,0.15,5
PtrNet-RSA,USNET,80,270,0.62,0.16,5
PtrNet-RSA,USNET,80,280,0.80,0.20,5
PtrNet-RSA,USNET,80,290,1.12,0.29,5
PtrNet-RSA,USNET,80,300,1.36,0.31,5
PtrNet-RSA,USNET,80,310,1.72,0.30,5
PtrNet-RSA,USNET,80,320,2.05,0.39,5
PtrNet-RSA,USNET,80,330,2.34,0.33,5
PtrNet-RSA,USNET,80,260,0.14,0.12,20
PtrNet-RSA,USNET,80,270,0.26,0.14,20
PtrNet-RSA,USNET,80,280,0.38,0.22,20
PtrNet-RSA,USNET,80,290,0.53,0.25,20
PtrNet-RSA,USNET,80,300,0.69,0.25,20
PtrNet-RSA,USNET,80,310,0.95,0.39,20
PtrNet-RSA,USNET,80,320,1.14,0.40,20
PtrNet-RSA,USNET,80,330,1.42,0.46,20
"""

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

csv_data_length = """
publication,topology,N_slots,load,mean,stddev
DeepRMSA,NSFNET,100,250,5.10,0
DeepRMSA,COST239,100,600,6.75,0
Reward-RMSA,NSFNET,100,168,1.00,0
Reward-RMSA,NSFNET,100,182,1.40,0
Reward-RMSA,NSFNET,100,196,2.10,0
Reward-RMSA,NSFNET,100,210,2.70,0
GCN-RMSA,NSFNET,100,154,0.67,0
GCN-RMSA,NSFNET,100,168,1.00,0
GCN-RMSA,NSFNET,100,182,1.42,0
GCN-RMSA,NSFNET,100,196,2.10,0
GCN-RMSA,NSFNET,100,210,2.76,0
GCN-RMSA,COST239,100,368,0.78,0
GCN-RMSA,COST239,100,391,1.30,0
GCN-RMSA,COST239,100,414,1.70,0
GCN-RMSA,COST239,100,437,2.30,0
GCN-RMSA,COST239,100,460,2.70,0
GCN-RMSA,USNET,100,320,0.75,0
GCN-RMSA,USNET,100,340,1.20,0
GCN-RMSA,USNET,100,360,1.50,0
GCN-RMSA,USNET,100,380,2.15,0
GCN-RMSA,USNET,100,400,2.70,0
MaskRSA,NSFNET,80,80,0.26,0
MaskRSA,NSFNET,80,90,0.60,0
MaskRSA,NSFNET,80,100,0.90,0
MaskRSA,NSFNET,80,110,1.80,0
MaskRSA,NSFNET,80,120,2.47,0
MaskRSA,NSFNET,80,130,3.00,0
MaskRSA,NSFNET,80,140,4.00,0
MaskRSA,NSFNET,80,150,5.00,0
MaskRSA,NSFNET,80,160,6.00,0
MaskRSA,JPN48,80,120,1.60,0
MaskRSA,JPN48,80,130,2.50,0
MaskRSA,JPN48,80,140,3.50,0
MaskRSA,JPN48,80,150,4.50,0
MaskRSA,JPN48,80,160,5.00,0
PtrNet-RSA,NSFNET,40,180,0.80,0
PtrNet-RSA,NSFNET,40,190,1.25,0
PtrNet-RSA,NSFNET,40,200,1.50,0
PtrNet-RSA,NSFNET,40,210,1.90,0
PtrNet-RSA,NSFNET,40,220,2.50,0
PtrNet-RSA,NSFNET,40,230,2.90,0
PtrNet-RSA,NSFNET,40,240,3.50,0
PtrNet-RSA,COST239,40,340,2.50,0
PtrNet-RSA,COST239,40,360,3.25,0
PtrNet-RSA,COST239,40,380,3.80,0
PtrNet-RSA,COST239,40,400,4.60,0
PtrNet-RSA,COST239,40,420,5.50,0
PtrNet-RSA,USNET,40,210,0.85,0
PtrNet-RSA,USNET,40,220,1.40,0
PtrNet-RSA,USNET,40,230,1.85,0
PtrNet-RSA,USNET,40,240,2.20,0
PtrNet-RSA,USNET,40,250,3.10,0
PtrNet-RSA,USNET,40,260,3.60,0
PtrNet-RSA,USNET,40,270,4.40,0
PtrNet-RSA,USNET,40,280,4.90,0
PtrNet-RSA,NSFNET,80,200,0.60,0
PtrNet-RSA,NSFNET,80,210,0.80,0
PtrNet-RSA,NSFNET,80,220,1.20,0
PtrNet-RSA,NSFNET,80,230,1.40,0
PtrNet-RSA,NSFNET,80,240,1.90,0
PtrNet-RSA,COST239,80,420,2.50,0
PtrNet-RSA,COST239,80,440,3.00,0
PtrNet-RSA,COST239,80,460,3.50,0
PtrNet-RSA,USNET,80,260,1.00,0
PtrNet-RSA,USNET,80,270,1.20,0
PtrNet-RSA,USNET,80,280,1.40,0
PtrNet-RSA,USNET,80,290,1.70,0
PtrNet-RSA,USNET,80,300,2.20,0
PtrNet-RSA,USNET,80,310,2.40,0
PtrNet-RSA,USNET,80,320,2.80,0
PtrNet-RSA,USNET,80,330,3.10,0
"""

csv_data_length_ours = """
publication,topology,N_slots,load,mean,stddev
DeepRMSA,NSFNET,100,250,5.13,0.32
DeepRMSA,COST239,100,600,6.69,0.35
Reward-RMSA,NSFNET,100,168,1.11,0.15
Reward-RMSA,NSFNET,100,182,1.59,0.22
Reward-RMSA,NSFNET,100,196,2.19,0.28
Reward-RMSA,NSFNET,100,210,2.87,0.29
GCN-RMSA,NSFNET,100,154,0.78,0.10
GCN-RMSA,NSFNET,100,168,1.11,0.15
GCN-RMSA,NSFNET,100,182,1.59,0.22
GCN-RMSA,NSFNET,100,196,2.19,0.28
GCN-RMSA,NSFNET,100,210,2.87,0.29
GCN-RMSA,COST239,100,368,0.78,0.21
GCN-RMSA,COST239,100,391,1.03,0.24
GCN-RMSA,COST239,100,414,1.46,0.27
GCN-RMSA,COST239,100,437,1.84,0.34
GCN-RMSA,COST239,100,460,2.39,0.31
GCN-RMSA,USNET,100,320,1.02,0.28
GCN-RMSA,USNET,100,340,1.27,0.33
GCN-RMSA,USNET,100,360,1.67,0.30
GCN-RMSA,USNET,100,380,2.25,0.43
GCN-RMSA,USNET,100,400,2.72,0.42
MaskRSA,NSFNET,80,80,0.33,0.04
MaskRSA,NSFNET,80,90,0.66,0.09
MaskRSA,NSFNET,80,100,1.08,0.10
MaskRSA,NSFNET,80,110,1.68,0.10
MaskRSA,NSFNET,80,120,2.31,0.18
MaskRSA,NSFNET,80,130,3.22,0.15
MaskRSA,NSFNET,80,140,4.09,0.19
MaskRSA,NSFNET,80,150,5.11,0.22
MaskRSA,NSFNET,80,160,6.25,0.25
MaskRSA,JPN48,80,120,1.91,0.21
MaskRSA,JPN48,80,130,2.80,0.33
MaskRSA,JPN48,80,140,3.63,0.33
MaskRSA,JPN48,80,150,4.56,0.31
MaskRSA,JPN48,80,160,5.48,0.41
PtrNet-RSA,NSFNET,40,180,0.91,0.28
PtrNet-RSA,NSFNET,40,190,1.23,0.28
PtrNet-RSA,NSFNET,40,200,1.67,0.31
PtrNet-RSA,NSFNET,40,210,1.98,0.34
PtrNet-RSA,NSFNET,40,220,2.47,0.43
PtrNet-RSA,NSFNET,40,230,2.99,0.48
PtrNet-RSA,NSFNET,40,240,3.81,0.71
PtrNet-RSA,COST239,40,340,3.07,0.37
PtrNet-RSA,COST239,40,360,3.60,0.43
PtrNet-RSA,COST239,40,380,4.26,0.51
PtrNet-RSA,COST239,40,400,4.92,0.52
PtrNet-RSA,COST239,40,420,5.73,0.71
PtrNet-RSA,USNET,40,210,1.01,0.16
PtrNet-RSA,USNET,40,220,1.44,0.22
PtrNet-RSA,USNET,40,230,1.93,0.29
PtrNet-RSA,USNET,40,240,2.54,0.35
PtrNet-RSA,USNET,40,250,3.16,0.43
PtrNet-RSA,USNET,40,260,3.89,0.49
PtrNet-RSA,USNET,40,270,4.53,0.42
PtrNet-RSA,USNET,40,280,5.40,0.46
PtrNet-RSA,NSFNET,80,200,0.55,0.17
PtrNet-RSA,NSFNET,80,210,0.74,0.15
PtrNet-RSA,NSFNET,80,220,1.00,0.23
PtrNet-RSA,NSFNET,80,230,1.17,0.23
PtrNet-RSA,NSFNET,80,240,1.40,0.28
PtrNet-RSA,COST239,80,420,2.33,0.40
PtrNet-RSA,COST239,80,440,2.83,0.38
PtrNet-RSA,COST239,80,460,3.16,0.51
PtrNet-RSA,USNET,80,260,1.21,0.22
PtrNet-RSA,USNET,80,270,1.46,0.23
PtrNet-RSA,USNET,80,280,1.72,0.29
PtrNet-RSA,USNET,80,290,2.07,0.26
PtrNet-RSA,USNET,80,300,2.41,0.33
PtrNet-RSA,USNET,80,310,2.78,0.34
PtrNet-RSA,USNET,80,320,3.12,0.33
PtrNet-RSA,USNET,80,330,3.46,0.34
"""


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

    data_file = '../../../../experiment_data/JOCN2024/experiment_results_eval.csv'
    df = pd.read_csv(data_file)
    # Filter to only have these columns: NAME,TOPOLOGY,LOAD,K,service_blocking_probability_mean/std/iqr_lower/iqr_upper
    df = df[['NAME', 'TOPOLOGY', 'LOAD', 'K', 'WEIGHT', 'service_blocking_probability_mean', 'service_blocking_probability_std',
                'service_blocking_probability_iqr_lower', 'service_blocking_probability_iqr_upper']]
    def get_n_slots(name):
        return 40 if name == 'PtrNet-RSA-40' else 80 if name in ['PtrNet-RSA-80', 'MaskRSA'] else 100
    def get_topology(name):
        return 'JPN48' if 'JPN48' in name.upper() else 'COST239' if 'COST239' in name.upper() else 'USNET' if 'USNET' in name.upper() else 'NSFNET'
    def get_publication(name):
        return 'PtrNet-RSA' if 'PtrNet-RSA' in name else name
    # Add in the number of slots
    df['N_slots'] = df['NAME'].apply(get_n_slots)
    # Add in the topology
    df['topology'] = df['TOPOLOGY'].apply(get_topology)
    # Add in the publication
    df['publication'] = df['NAME'].apply(get_publication)
    # Rename columns to match publication,topology,N_slots,load,mean,stddev
    df = df.rename(columns={'LOAD': 'load', 'K': 'k',
                            'service_blocking_probability_mean': 'mean',
                            'service_blocking_probability_std': 'stddev',
                            'service_blocking_probability_iqr_lower': 'iqr_lower',
                            'service_blocking_probability_iqr_upper': 'iqr_upper'})
    # Multiply mean, stddev, iqr_lower, iqr_upper by 100 to get percentages
    # Add 0.0001 to values to avoid zeros and ensure plotting
    df['mean'] *= 100
    df['stddev'] *= 100
    df['iqr_lower'] *= 100
    df['iqr_upper'] *= 100
    # if IQR upper is zero, set it to df['mean']
    mask_mean = df[df['iqr_upper'] == 0]['mean']
    mask_stddev = df[df['iqr_upper'] == 0]['stddev']
    df.loc[mask_mean.index, 'iqr_upper'] = mask_mean
    df.loc[mask_mean.index, 'iqr_lower'] = mask_mean - mask_stddev

    # Filter by weight=="" to get df_hops
    df_hops = df[df['WEIGHT'] != '--weight=weight']

    # Filter by weight=='--weight' to get df_length_ours
    df_length_ours = df[df['WEIGHT'] == '--weight=weight']

    # Read the CSV data
    #df_hops = pd.read_csv(StringIO(csv_data_hops))
    df_rl = pd.read_csv(StringIO(csv_data_rl))
    df_length = pd.read_csv(StringIO(csv_data_length))
    #df_length_ours = pd.read_csv(StringIO(csv_data_length_ours))

    # Define the publications and topologies
    publications = ['DeepRMSA', 'Reward-RMSA', 'GCN-RMSA', 'MaskRSA', 'PtrNet-RSA-40', 'PtrNet-RSA-80']
    topologies = ['NSFNET', 'COST239', 'USNET', 'JPN48']

    # red
    rl_col = '#d62728'
    # blue
    length_col = '#1f77b4'
    # green
    length_ours_col = '#2ca02c'
    # orange
    hops_5_col = '#ff7f0e'
    # purple
    hops_50_col = '#9467bd'

    # Function to plot data for a single case
    def plot_case(ax, pub, topology, n_slots):
        pub_filter = 'PtrNet-RSA' if 'PtrNet-RSA' in pub else pub
        case_hops = df_hops[
            (df_hops['publication'] == pub_filter) & (df_hops['topology'] == topology) & (df_hops['N_slots'] == n_slots)]
        case_rl = df_rl[(df_rl['publication'] == pub_filter) & (df_rl['topology'] == topology) & (df_rl['N_slots'] == n_slots)]
        case_length = df_length[
            (df_length['publication'] == pub_filter) & (df_length['topology'] == topology) & (df_length['N_slots'] == n_slots)]
        case_length_ours = df_length_ours[
            (df_length_ours['publication'] == pub_filter) & (df_length_ours['topology'] == topology) & (df_length_ours['N_slots'] == n_slots)]

        if case_length.empty:
            return False, [], []

        lines = []
        labels = []

        # Plot data for DeepRMSA (just dot plot)
        if pub == "DeepRMSA":
            # Use errorbars for DeepRMSA as just single points
            line1 = ax.errorbar(case_rl['load'], case_rl['mean'], yerr=case_rl['stddev'],
                                label='RL', marker='x', capsize=5, color=rl_col, linewidth=3, markersize=20)
            line2 = ax.errorbar(case_length['load'], case_length['mean'], yerr=case_length['stddev'],
                                label='5-SP-FF$^{published}_{km}$', marker='x', capsize=5, color=length_col, linewidth=3, markersize=20)
            line3 = ax.errorbar(case_length_ours['load'], case_length_ours['mean'], yerr=case_length_ours['stddev'],
                                label='5-SP-FF$_{km}$', marker='o', capsize=5, color=length_ours_col, linewidth=3, markersize=10)
            case_hops_5 = case_hops[case_hops['k'] == 5]
            case_hops_50 = case_hops[case_hops['k'] == 50]
            line4 = ax.errorbar(case_hops_5['load'], case_hops_5['mean'], yerr=case_hops_5['stddev'],
                                label='5-SP-FF$_{hops}$', marker='o', capsize=5, color=hops_5_col, linewidth=3, markersize=10)
            line5 = ax.errorbar(case_hops_50['load'], case_hops_50['mean'], yerr=case_hops_50['stddev'],
                                label='5-SP-FF$_{hops}$', marker='o', capsize=5, color=hops_50_col, linewidth=3,
                                markersize=10)
            #lines.extend([line1, line2, line3, line4])
            #labels.extend(['RL', '5-SP-FF$_{published}$', '5-SP-FF$_{hops}$', '5-SP-FF$_{ours}$'])

        else:

            # Plot RL data
            if not case_rl.empty:
                line = ax.plot(case_rl['load'], case_rl['mean'], label='RL', marker='x', linestyle='-', color=rl_col,
                               markersize=15, linewidth=2)
                lines.append(line[0])
                labels.append('RL')

            # Plot length data
            if not case_length.empty:
                line = ax.plot(case_length['load'], case_length['mean'], label='5-SP-FF$_{published}$', marker='x',
                                 markerfacecolor=length_col, linestyle='-', color=length_col, markersize=15)
                lines.append(line[0])
                labels.append('5-SP-FF$^{published}_{km}$')

            # Plot length data for our method
            if not case_length_ours.empty:
                line = ax.plot(case_length_ours['load'], case_length_ours['mean'], label='5-SP-FF$_{ours}$', marker='o',
                               markerfacecolor=length_ours_col, linestyle='-', color=length_ours_col)
                ax.fill_between(case_length_ours['load'], case_length_ours['mean'] - case_length_ours['stddev'],
                                case_length_ours['mean'] + case_length_ours['stddev'], alpha=0.2, color=length_ours_col)
                lines.append(line[0])
                labels.append('5-SP-FF$_{km}$')

            # Plot hops data for k=5 and k=50 if available
            for k in [5, 50]:
                case_hops_k = case_hops[case_hops['k'] == k]
                if not case_hops_k.empty:
                    line = ax.plot(case_hops_k['load'], case_hops_k['mean'], label=f'{k}'+'-SP-FF$_{hops}$',
                                   marker='o', linestyle='-', color=hops_5_col if k == 5 else hops_50_col)
                    # Do fillbetween for error region
                    ax.fill_between(case_hops_k['load'], case_hops_k['mean'] - case_hops_k['stddev'],
                                    case_hops_k['mean'] + case_hops_k['stddev'], alpha=0.2,
                                    color=hops_5_col if k == 5 else hops_50_col)
                    lines.append(line[0])
                    labels.append(f'{k}'+'-SP-FF$_{hops}$')

        title = f'{pub}\n\n{topology}' if topology == 'NSFNET' else topology
        ax.set_title(title, fontsize=32)
        return True, lines, labels


    # Define grid
    grid = [
        ['NSFNET', 'COST239'],
        ['NSFNET'],
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

            # make y-axis log scale
            if publication == 'DeepRMSA':
                ax.set_yticks(np.arange(0, 10, 1))
                # show grid for y-axis
                ax.yaxis.grid(True)
                ax.set_xticks(np.arange(0, 1000, 50))
            else:
                ax.set_yscale('log')
                # show grid for y-axis
                ax.yaxis.grid(True)
                ax.set_xticks(np.arange(0, 1000, 25))
                if publication == 'PtrNet-RSA-40' and topology == 'COST239':
                    ax.set_ylim(0.0005, 9.9)

            # Plot the case and check if data was plotted
            data_plotted, lines, labels = plot_case(ax, publication, topology, n_slots)

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
