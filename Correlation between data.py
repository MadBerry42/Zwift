import numpy as np

aerobic = [3, 4, 5]
anaerobic = [1, 2, 4]

R = np.corrcoef(aerobic, anaerobic)
r_pearson = R[0, 1]

# print(R)
print("Pearson's coefficient is: ", r_pearson)

