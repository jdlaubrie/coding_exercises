# python3
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def fcirc(x, c10, k1, k2):
    kappa = 0.
    beta1 = 45.0*np.pi/180.0
    beta2 = -45.0*np.pi/180.0
    stress = 2.*c10*(x**2 - 1./x**4) + \
             2.*k1*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))*\
             np.exp(k2*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))**2)*\
             (kappa*(x**2 - 1./x**4)+(1.-3.*kappa)*(x*np.cos(beta1))**2) + \
             2.*k1*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))*\
             np.exp(k2*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))**2)*\
             (kappa*(x**2 - 1./x**4)+(1.-3.*kappa)*(x*np.cos(beta2))**2)
    return stress
    
def flong(x, c10, k1, k2):
    kappa = 0.
    beta1 = 45.0*np.pi/180.0
    beta2 = -45.0*np.pi/180.0
    stress = 2.*c10*(x**2 - 1./x**4) + \
             2.*k1*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))*\
             np.exp(k2*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))**2)*\
             (kappa*(x**2 - 1./x**4)+(1.-3.*kappa)*(x*np.sin(beta1))**2) + \
             2.*k1*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))*\
             np.exp(k2*(kappa*(2.*x**2+1./x**4-3.)+(1.-3.*kappa)*(x**2-1.0))**2)*\
             (kappa*(x**2 - 1./x**4)+(1.-3.*kappa)*(x*np.sin(beta2))**2)
    return stress

#--------------------------------------------------------------------------
#          experimental data
#--------------------------------------------------------------------------
import pandas as pd
file_data = open('peuu_R000V03.dat', 'r')
data_lines = file_data.readlines()
data_frame = pd.Series(data_lines)
data_frame = data_frame.replace(r'\n','', regex=True)
data_frame = data_frame.replace(r'\r\n','', regex=True)
data_frame = data_frame.replace(r'\r','', regex=True)
#--------------------------------------------------------------------------
for line in data_frame:
    tmp_df = data_frame.str.strip()
    tmp_df = tmp_df.str.split('  ',expand=True)
    np.array(tmp_df.values, dtype=float)
    exp_data = np.array(tmp_df.values, dtype=float)

#--------------------------------------------------------------------------
#          curves fitting
#--------------------------------------------------------------------------
ndata = 10
# stretch lambda_1 and lambda_2
xdata = exp_data[:ndata,0]
# bulge stress
ydata = exp_data[:ndata,1]*xdata**2

# 1st fit is with long
popt1, pcov1 = curve_fit(flong, xdata, ydata,
    p0=[563.9, 1.0, 1.0], 
    #bounds=([0.0],[np.inf]))
    bounds=([100.0, 0.0, 0.0],[np.inf, np.inf, np.inf]))
print(popt1)
# 2nd fit is with circ
popt2, pcov2 = curve_fit(fcirc, xdata, ydata,
    p0=popt1, 
    #bounds=([0.0],[np.inf]))
    bounds=([100.0, 0.0, 0.0],[np.inf, np.inf, np.inf]))
print(popt2)

from sklearn.metrics import r2_score
y_true = ydata
# longitudinal
y1_pred = flong(xdata, *popt2)
deter1 = r2_score(y_true, y1_pred)
print("Longitudinal determination: {}".format(deter1))
# circumferential
y2_pred = fcirc(xdata, *popt2)
deter2 = r2_score(y_true, y2_pred)
print("Circumferential determination: {}".format(deter2))

#--------------------------------------------------------------------------
#          fem simulation results
#--------------------------------------------------------------------------
import pandas as pd
file_data = open('fem_data.dat', 'r')
data_lines = file_data.readlines()
data_frame = pd.Series(data_lines)
data_frame = data_frame.replace(r'\n','', regex=True)
data_frame = data_frame.replace(r'\r\n','', regex=True)
data_frame = data_frame.replace(r'\r','', regex=True)
#--------------------------------------------------------------------------
for line in data_frame:
    tmp_df = data_frame.str.strip()
    tmp_df = tmp_df.str.split(' ',expand=True)
    np.array(tmp_df.values, dtype=float)
    fem_data = np.array(tmp_df.values, dtype=float)

xfem = 1.0 + fem_data[:,1]/3.0
yfem = 1000.0*fem_data[:,3]

#--------------------------------------------------------------------------
#          curves visualization. graphics
#--------------------------------------------------------------------------
xfit = np.linspace(1.0,1.07,num=15)
fig, ax = plt.subplots(constrained_layout=True)

ax.plot(xfit, flong(xfit, *popt2), 'b-', label='Longitudinal fit')
ax.plot(xfit, fcirc(xfit, *popt2), 'r-', label='Circumferential fit')
ax.plot(xfem, yfem, 'g-', label='Abaqus simulation')
ax.plot(exp_data[:,0], exp_data[:,1]*(exp_data[:,0])**2, 'o')
ax.plot(xdata, ydata, 'o', label='Experimental data')

ax.set_xlabel('stretch [-]',fontsize=14)
ax.set_ylabel('stress [kPa]',fontsize=14)
ax.set_ylim(bottom=0,top=500)
ax.set_xlim(left=1.0,right=1.1)
for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    label.set_fontsize(14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left',fontsize=14)

plt.show

FIGURENAME = 'fitting_py.eps'
plt.savefig(FIGURENAME)

plt.close('all')

#[ 100.         1017.0845917     9.36312794]
#Longitudinal determination: 0.9724576302156388
#Circumferential determination: 0.9724576302156389

