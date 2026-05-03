import pandas as pd
import matplotlib.pyplot as plt
import io

# Data as CSV string with all measurements
data_str = """launch_power,service_blocking,service_blocking_err,bitrate_blocking,bitrate_blocking_err,accepted_services,accepted_services_err,accepted_bitrate,accepted_bitrate_err,utilisation,utilisation_err
0.0,0.07870,0.02941,0.11200,0.03884,921,29,670940,29345,0.18,0.02
0.1,0.07460,0.03337,0.10668,0.03874,925,33,674960,29269,0.19,0.02
0.2,0.07640,0.03399,0.10570,0.04005,924,34,675700,30258,0.19,0.02
0.3,0.05810,0.00495,0.08693,0.01821,941,5,689880,13762,0.19,0.01
0.4,0.05980,0.00637,0.08722,0.01811,940,6,689660,13684,0.19,0.02
0.5,0.05810,0.00495,0.08550,0.01785,942,5,690960,13488,0.19,0.02
0.6,0.06470,0.01737,0.09098,0.02901,935,17,686820,21921,0.19,0.02
0.7,0.08010,0.04338,0.10480,0.04620,920,43,676380,34907,0.19,0.02
0.8,0.06040,0.01711,0.08280,0.02127,940,17,693000,16074,0.20,0.02
0.9,0.08310,0.08909,0.10149,0.09167,917,89,678880,69261,0.20,0.02
1.0,0.11500,0.14969,0.13394,0.14810,885,150,654360,111895,0.19,0.03"""


# Rest of the plotting code remains the same
def create_plot(as_line=False, fontsize=12):
    plt.figure(figsize=(10, 6))

    if as_line:
        plt.plot(df['launch_power'], df['service_blocking'] * 100, 'b-o')
    else:
        plt.scatter(df['launch_power'], df['service_blocking'] * 100, color='blue')

    # Add error bars
    plt.errorbar(df['launch_power'],
                 df['service_blocking'] * 100,
                 yerr=df['service_blocking_err'] * 100,
                 fmt='none',  # Don't add additional markers
                 capsize=5,  # Add caps to error bars
                 color='blue',
                 alpha=0.5)  # Make error bars slightly transparent

    # Customize the plot
    plt.xlabel('Launch Power (dBm)', fontsize=fontsize)
    plt.ylabel('Service Blocking Probability (%)', fontsize=fontsize)

    # Increase tick label size
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return plt


if __name__ == '__main__':
    # Read the CSV string into a DataFrame
    df = pd.read_csv(io.StringIO(data_str))

    # Create and show the scatter plot with default font size
    plot = create_plot(as_line=False, fontsize=12)
    plt.show()

    # Example of creating a line plot with larger font size
    # Uncomment the following lines to create alternative versions
    # plot = create_plot(as_line=True, fontsize=14)
    # plt.show()
