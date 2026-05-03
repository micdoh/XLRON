import pandas as pd
import matplotlib.pyplot as plt


def read_csv(file_path):
    return pd.read_csv(file_path, skiprows=1, names=['Step', 'Value', 'Min', 'Max'])


def plot_data(mean_train, iqr_lower_train, iqr_upper_train, mean_ksplf, iqr_lower_ksplf, iqr_upper_ksplf):
    plt.figure(figsize=(4,3))

    plt.plot(mean_train['Step'], mean_train['Value'], label='RL')
    plt.fill_between(mean_train['Step'],
                     mean_train['Value_upper'],
                     mean_train['Value_lower'],
                     alpha=0.3,)

    plt.plot(mean_ksplf['Step'], mean_ksplf['Value'], label='0.5dBm fixed')
    plt.fill_between(mean_ksplf['Step'],
                        mean_ksplf['Value_upper'],
                        mean_ksplf['Value_lower'],
                        alpha=0.3,)

    plt.xlabel('Traffic Request (per parallel env)')
    plt.ylabel('Bitrate Blocking Probability')
    plt.title('')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.yscale('log')
    plt.ylim(1e-4, 2e-1)
    plt.xlim(-mean_train['Step'].max()*0.01, mean_train['Step'].max()*1.01)
    # Add some space to rightside so that final tick is not cut off
    plt.subplots_adjust(right=0.9)
    plt.tight_layout()
    plt.savefig("/Users/michaeldoherty/Desktop/CDT/OneDrive - University College London/Publications/OFC2025/training.png")
    plt.show()


def main():
    # File paths (adjust these to match your actual file names)
    mean_file_train = '../../data/OFC2025/bbp_mean.csv'
    iqr_lower_file_train = '../../data/OFC2025/bbp_iqr_lower.csv'
    iqr_upper_file_train = '../../data/OFC2025/bbp_iqr_upper.csv'
    mean_file_ksplf = '../../data/OFC2025/bbp_ksplf_mean.csv'
    iqr_lower_file_ksplf = '../../data/OFC2025/bbp_ksplf_iqr_lower.csv'
    iqr_upper_file_ksplf = '../../data/OFC2025/bbp_ksplf_iqr_upper.csv'
    
    # Read CSV files
    mean_train = read_csv(mean_file_train)
    iqr_lower_train = read_csv(iqr_lower_file_train)
    iqr_upper_train = read_csv(iqr_upper_file_train)
    mean_ksplf = read_csv(mean_file_ksplf)
    iqr_lower_ksplf = read_csv(iqr_lower_file_ksplf)
    iqr_upper_ksplf = read_csv(iqr_upper_file_ksplf)

    # Join iqr onto mean
    mean_train = mean_train.join(iqr_lower_train, rsuffix='_lower')
    mean_train = mean_train.join(iqr_upper_train, rsuffix='_upper')
    mean_ksplf = mean_ksplf.join(iqr_lower_ksplf, rsuffix='_lower')
    mean_ksplf = mean_ksplf.join(iqr_upper_ksplf, rsuffix='_upper')
    # Fill nans with earlier value
    mean_train['Value_lower'] = mean_train['Value_lower'].ffill()
    mean_train['Value_upper'] = mean_train['Value_upper'].ffill()
    mean_ksplf['Value_lower'] = mean_ksplf['Value_lower'].ffill()
    mean_ksplf['Value_upper'] = mean_ksplf['Value_upper'].ffill()
    mean_ksplf['Step'] = range(len(mean_ksplf))
    # Make mean_train step start from 0
    mean_train['Step'] = mean_train['Step'] - mean_train['Step'].min()

    # Plot data
    plot_data(mean_train, iqr_lower_train, iqr_upper_train, mean_ksplf, iqr_lower_ksplf, iqr_upper_ksplf)


if __name__ == "__main__":
    main()
