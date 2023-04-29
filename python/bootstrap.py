from scipy.stats import t, norm, chi2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

def func2fit(x, c10, c20, c30):
    I1 = x**2 + 2./x
    y = (2.0*c10 + 4.0*c20*(I1-3.0) + \
            6.0*c30*((I1-3.0)**2))*(x**2 - 1.0/x)
    return y

x = np.array([1.041, 1.059, 1.098, 1.119, 1.131, 1.140, 1.148, 1.153, 1.157,
        1.162, 1.166], dtype=np.float64)
y = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0,
        140.0, 160.0, 180.0, 200.0], dtype=np.float64)

# fitting. func, data(x,y), guess, bounds
popt, pcov = curve_fit(func2fit, x, y, p0=[10.0,10.0,10.0])
print("parameters = \n{}".format(popt))
print("covariance of parameters = \n{}".format(pcov))

#compute the determination coefficient with ML library
from sklearn.metrics import r2_score
deter = r2_score(y, func2fit(x, *popt))
print("Determination Coefficient: {}".format(deter))

# original data
res = y - func2fit(x, *popt)
ndata = res.shape[0]

# bootstrap samples
nboot = 100
boot_res = np.random.choice(res,(nboot,ndata))
boot_y = np.zeros((nboot,ndata),dtype=np.float64)
boot_opt = np.zeros((nboot,popt.shape[0]),dtype=np.float64)
for i in range(nboot):
    boot_y[i,:] = func2fit(x,*popt) + boot_res[i,:]
    boot_opt[i,:], _ = curve_fit(func2fit, x, boot_y[i,:], p0=[10.0,10.0,10.0])

boot_mean = np.mean(boot_opt,axis=0)
boot_std = np.std(boot_opt, axis=0, ddof=1)
boot_cov = np.cov(boot_opt.transpose())

print("Bootstrap mean = {}".format(boot_mean))
print("Bootstrap std = {}".format(boot_std))

confidence = 0.95

z = np.linspace(boot_mean-3.*boot_std,boot_mean+3.*boot_std,100)
dist0 = norm(loc=boot_mean, scale=boot_std)

p_value0 = dist0.cdf(popt)
interval0 = dist0.interval(confidence)

# boot_mean + tstudent(confidence)*boot_std
print("\nSeparate confidence interval = \n{}".format(interval0))
print("The p-values are = {}".format(p_value0))

p = 3
dist3 = chi2(p)
diff = boot_mean - popt
boot_cov_inv = np.linalg.inv(boot_cov)
z_square = np.einsum('i,ij,j',diff,boot_cov_inv,diff)
print("\nZ-squared (multivariate test) = {}".format(z_square))
print("chi-squared at 0.475 confidence = {}".format(dist3.ppf(0.475)))
#print("P-value of test = {}".format(dist3.cdf(z_square)))

factor = nboot*boot_std/np.sqrt(ndata)
#------------------------------------------------------------------------------#
fig, ax = plt.subplots(2,3, figsize=plt.figaspect(0.5))

# data distribution
ax[0,0].hist(boot_opt[:,0], bins=15)
ax[0,0].plot(z[:,0], factor[0]*dist0.pdf(z)[:,0])
# QQplot for error 1
sm.graphics.qqplot(boot_opt[:,0], line='s', ax=ax[1,0])

# data distribution
ax[0,1].hist(boot_opt[:,1], bins=15)
ax[0,1].plot(z[:,1], factor[1]*dist0.pdf(z)[:,1])
# QQplot for error 1
sm.graphics.qqplot(boot_opt[:,1], line='s', ax=ax[1,1])

# data distribution
ax[0,2].hist(boot_opt[:,2], bins=15)
ax[0,2].plot(z[:,2], factor[2]*dist0.pdf(z)[:,2])
# QQplot for error 1
sm.graphics.qqplot(boot_opt[:,2], line='s', ax=ax[1,2])

# make the graphic nicer
#ax.set_xlabel('stretch [-]',fontsize=14)
#ax.set_ylabel('stress [kPa]',fontsize=14)
#ax.set_ylim(bottom=0,top=500)
#ax.set_xlim(left=1.0,right=1.1)
#open the box and add legend

fig.tight_layout()
plt.show
#file output
FIGURENAME = 'bootstrap.png'
plt.savefig(FIGURENAME)
#close graphical tools
plt.close('all')

