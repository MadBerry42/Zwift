import pandas as pd
import numpy as np

# Importing data
path = ""
n_subjects = 10 # Count the number of folders 
RPE = np.array[12, 14, 15]
for i in range(1, n_subjects):
    P1 = np.zeros(3, 2)
    if i < 10:
        ID = f"00{i}"
    if i >= 10:
        ID = f"0{i}"
    

    power_hc = pd.read_csv(f"path\{ID}_protocol_handcycle", usecols=['Power'])
    power_bc = pd.read_csv(f"path\{ID}_protocol_bycycle", usecols=['Power'])

    # Convert dataframe into a numpy array for better handling
    power_hc = np.array(power_hc)
    power_bc = np.array(power_bc)


    # Isolating the blocks with a certain RPE
    hc_RPE_12 = power_hc[300 : 479]
    hc_RPE_14 = power_hc[480 : 659]
    hc_RPE_15 = power_hc[660 : 839]

    bc_RPE_12 = power_bc[300 : 479]
    bc_RPE_14 = power_bc[480 : 659]
    bc_RPE_15 = power_bc[660 : 839]

    P1[1, 1] = np.mean(hc_RPE_12)
    P1[2, 1] = np.mean(hc_RPE_14)
    P1[3, 1] = np.mean(hc_RPE_15)

    P1[1, 2] = np.mean(bc_RPE_12)
    P1[2, 2] = np.mean(bc_RPE_14)
    P1[3, 2] = np.mean(bc_RPE_15) 

    print(P1)    
