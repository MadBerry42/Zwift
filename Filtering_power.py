import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Import data and select the portion with fixed RPE
ID = 0
if ID < 10:
    ID = f"00{ID}"
else:
    ID = f"0{ID}"

data = pd.read_csv(f"C:\\Users\\maddalb\\Desktop\\git\\Zwift\\Acquisitions\\Protocol\\{ID}\\Zwift\\{ID}_handcycle_protocol.csv", usecols = ["Power"])
data = data[300: 839]
# Find cutting frequency from the PSD 
