# ptyhon3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv('data_frame.csv', sep=';')
exp_data = np.array(data_frame.values, dtype=float)

print(data_frame)

fig, ax = plt.subplots(1, 1)
ax.plot(exp_data[:,5], exp_data[:,1])
plt.show()

