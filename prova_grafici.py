import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ID = 3
setup = "handcycle"
limb = "wrist"
test = "protocol"

fig, (ax1, ax2) = plt.subplots(2)
processed_1 = pd.read_csv(f"00{ID}\\Processed Data\\00{ID}_{setup}_crank_{test}_processed.csv")
processed_2 = pd.read_csv(f"00{ID}\\Processed Data\\00{ID}_{setup}_{limb}_{test}_processed.csv")

processed_1 = processed_1.to_numpy()
processed_2 = processed_2.to_numpy()

timestamps_1 = processed_1[:, 0]
timestamps_2 = processed_2[:, 0]

print("Timestamps are: ", type(timestamps_1))

ax1.plot(timestamps_1, processed_1[:, 1])
ax2.plot(timestamps_2, processed_2[:, 1])

plt.show()
