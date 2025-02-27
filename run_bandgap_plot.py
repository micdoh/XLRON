import matplotlib.pyplot as plt
import numpy as np


def plot_vertical_multiband_array():
    rows = 5
    cols = 11
    full_array = np.zeros((rows, cols))

    full_array[:, :5] = 0
    full_array[:, 5:6] = -1
    full_array[:, 6:] = 0

    plt.figure(figsize=(10, 4))
    plt.imshow(full_array, cmap='Pastel1')

    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, int(full_array[i, j]), ha='center', va='center')

    plt.title('Link Slot Array')
    plt.yticks([])
    plt.xticks([2, 5, 8], ['C Band', 'Gap', 'L Band'])

    # Add only horizontal grid lines
    plt.gca().set_yticks(np.arange(-.5, rows, 1), minor=True)
    plt.grid(True, which='minor', axis='y', color='black', linewidth=1)

    plt.show()


plot_vertical_multiband_array()