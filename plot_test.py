import matplotlib.pyplot as plt
import numpy as np

# Data
traffic_loads = [250, 350, 400, 550, 700, 850, 1000]
multi_band_blocking = [-2.7e-10, 0.000199, 0.002108, 0.0502, 0.1237, 0.1917, 0.2515]
single_band_blocking = [0.59, 0.67, 0.70, 0.75, 0.80, 0.82, 0.84]

# Create plot with log scale y-axis
plt.figure(figsize=(10, 6))
plt.semilogy(traffic_loads, single_band_blocking, 'bo-', linewidth=2, markersize=8, label='Single Band')
plt.semilogy(traffic_loads, multi_band_blocking, 'ro-', linewidth=2, markersize=8, label='Multi Band')


plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel("Traffic Load (Erlangs)", fontsize=12)
plt.ylabel("Blocking Probability", fontsize=12)
plt.title("Single Band vs Multi Band Performance", fontsize=14)
#plt.ylim([1e-10, 1])
plt.yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
plt.legend()

plt.tight_layout()
plt.show()