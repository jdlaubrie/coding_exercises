# python3
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func_stress(x, c10):
    stress = 2.*c10*(x**2 - 1./x)
    return stress
    
#--------------------------------------------------------------------------
#          experimental data
#--------------------------------------------------------------------------
import pandas as pd
file_data = open('polymer_data.dat', 'r')   #open data file
data_lines = file_data.readlines()          #read lines of the file
data_frame = pd.Series(data_lines)          #give the lines to pandas
data_frame = data_frame.replace(r'\n','', regex=True)   #remove end-lines
data_frame = data_frame.replace(r'\r\n','', regex=True)
data_frame = data_frame.replace(r'\r','', regex=True)
#--------------------------------------------------------------------------
for line in data_frame:
    tmp_df = data_frame.str.strip()             #remove leading and trailing char.
    tmp_df = tmp_df.str.split('  ',expand=True) #split columns
    np.array(tmp_df.values, dtype=float)        #create an array from the data
    exp_data = np.array(tmp_df.values, dtype=float)

#--------------------------------------------------------------------------
#          curves fitting
#--------------------------------------------------------------------------
ndata = exp_data.shape[0]
# stretch lambda_1
xdata = exp_data[:ndata,0]
# stress sigma_1
ydata = exp_data[:ndata,1]*xdata**2

# fitting. func, data(x,y), guess, bounds
popt, pcov = curve_fit(func_stress, xdata, ydata,
    p0=[500.0], bounds=([50.0],[np.inf]))
print("C10 = {}".format(popt[0]))

#compute the determination coefficient with ML library
from sklearn.metrics import r2_score
y_true = ydata
y_pred = func_stress(xdata, *popt)
deter = r2_score(y_true, y_pred)
print("Determination Coefficient: {}".format(deter*100.0))

#--------------------------------------------------------------------------
#          curves visualization. graphics
#--------------------------------------------------------------------------
xfit = np.linspace(1.0,1.07,num=15)
fig, ax = plt.subplots(constrained_layout=True)

# plot curve and data
ax.plot(xfit, func_stress(xfit, *popt), 'b-', label='Fit')
ax.plot(xdata, ydata, 'o', label='Experimental data')

# make the graphic nicer
ax.set_xlabel('stretch [-]',fontsize=14)
ax.set_ylabel('stress [kPa]',fontsize=14)
ax.set_ylim(bottom=0,top=500)
ax.set_xlim(left=1.0,right=1.1)
for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    label.set_fontsize(14)
#open the box and add legend
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left',fontsize=14)

plt.show
#file output
FIGURENAME = 'fitting_py.eps'
plt.savefig(FIGURENAME)
#close graphical tools
plt.close('all')

