"""
The aim of this code is to define a confidence region through the bootstraping
method. The region is around the parameters resulting from non-linear fitting.

Made by JD Laubrie
"""

from scipy.stats import norm, chi2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

# objective function to fit
def func2fit(x, c10, c01):
    # this is a Mooney-Rivlin function for uniaxial stress
    I1 = x**2 + 2./x
    y = 2.0*c10*( x**2 - 1.0/x ) + \
        2.0*c01*( I1*(x**2 - 1.0/x) - (x**4 - 1.0/x**2) )
    return y

def Fitting(x, y):
    # fitting (optimization). func2fit, data(x,y), initial guess
    popt, pcov = curve_fit(func2fit, x, y, p0=[10.0,10.0])
    print("parameters = \n{}".format(popt))

    #compute the determination coefficient with ML library
    # if bigger than 0.95, the fitting is good
    from sklearn.metrics import r2_score
    deter = r2_score(y, func2fit(x, *popt))
    print("Determination Coefficient: {}".format(deter))

    # over-parametrization assessment
    nparam = popt.shape[0]
    # compute correlation matrix. corr_ij = cov_ij/(cov_ii*cov_jj)
    correlation = np.zeros((nparam,nparam),dtype=np.float64)
    for i in range(nparam):
        for j in range(nparam):
            correlation[i,j] = pcov[i,j]/\
                              np.sqrt(pcov[i,i]*pcov[j,j])

    # determinant of correlation measure dependence between parameters
    determinant = np.linalg.det(correlation)
    if determinant < 1.0e-4:
        is_overparameter = True
    else:
        is_overparameter = False
    print("Correlation matrix = \n{}".format(correlation))
    print("Over-parameterization: {},".format(is_overparameter) + \
          " det(R) = {}".format(determinant))

    return popt

def Bootstraping(x, y, residual, param, around_func=True):

    ndata = x.shape[0]
    nparam = param.shape[0]
    conf = 0.95
    nboot = 100

    # bootstrap samples
    boot_res = np.random.choice(residual,(nboot,ndata))
    boot_y = np.zeros((nboot,ndata),dtype=np.float64)
    boot_opt = np.zeros((nboot,nparam),dtype=np.float64)
    if around_func:
        for i in range(nboot):
            boot_y[i,:] = boot_res[i,:] + func2fit(x,*param)
            boot_opt[i,:], _ = curve_fit(func2fit, x, boot_y[i,:],
                                         p0=[10.0,10.0])
    else:
        for i in range(nboot):
            boot_y[i,:] = boot_res[i,:] + y
            boot_opt[i,:], _ = curve_fit(func2fit, x, boot_y[i,:],
                                         p0=[10.0,10.0])

    boot_mean = np.mean(boot_opt,axis=0)
    boot_std = np.std(boot_opt, axis=0, ddof=1)
    boot_cov = np.cov(boot_opt.transpose())

    print("\nBootstrap mean = {}".format(boot_mean))
    print("Bootstrap std = {}".format(boot_std))

    z = np.linspace(boot_mean-3.*boot_std,boot_mean+3.*boot_std,100)

    # assumed normal distribution for bootstrap data
    dist0 = norm(loc=boot_mean, scale=boot_std)
    interval0 = dist0.interval(conf)
    p_value0 = dist0.cdf(param)

    # boot_mean + norm(conf)*boot_std
    print("\nSeparate confidence interval ({}) ".format(conf*100.) +\
          "= \n{}".format(interval0))
    print("The p-values are = {}".format(p_value0))

    # multivariate test
    dist3 = chi2(nparam)
    diff = boot_mean - param
    boot_cov_inv = np.linalg.inv(boot_cov)
    z_square = np.einsum('i,ij,j',diff,boot_cov_inv,diff)
    print("\nZ-squared (multivariate test) = {}".format(z_square))
    print("chi-squared at {} ".format(100.*conf/2.) +\
          "= {}".format(dist3.ppf(conf/2.)) )
    #print("P-value of test = {}".format(dist3.cdf(z_square)))

#------------------------------------------------------------------------------#
    factor = nboot*boot_std/np.sqrt(ndata)
    fig, ax = plt.subplots(2,2, figsize=plt.figaspect(0.5))

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

    fig.tight_layout()
    plt.show
    FIGURENAME = 'bootstrap.png'
    plt.savefig(FIGURENAME)
    plt.close('all')

#------------------------------------------------------------------------------#
    fig, ax = plt.subplots()

    # data distribution
    for i in range(nboot):
        ax.scatter(x, boot_y[i,:])
    ax.plot(x, y, marker='*', color='red', label='exp')
    ax.plot(x, func2fit(x, *param), label='fit')
    ax.legend(loc='upper left')

    fig.tight_layout()
    plt.show
    FIGURENAME = 'bootstrap_fit.png'
    plt.savefig(FIGURENAME)
    plt.close('all')

def ResidualAnalysis(x, y, param):

    # compute the residuals between experiment and model
    residual = y - func2fit(x, *param)

    fig, ax = plt.subplots(1,2, figsize=plt.figaspect(0.5))

    # data distribution
    ax[0].scatter(x, residual)

    # QQplot for error 1
    sm.graphics.qqplot(residual, line='s', ax=ax[1])

    fig.tight_layout()
    plt.show
    FIGURENAME = 'bootstrap_res.png'
    plt.savefig(FIGURENAME)
    plt.close('all')

    return residual
#==============================================================================#
# experimental data, univariate and univariable. y=f(x)
x = np.array([1.041, 1.059, 1.098, 1.119, 1.131, 1.140, 1.148, 1.153, 1.157,
        1.162, 1.166], dtype=np.float64)
y = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0,
        140.0, 160.0, 180.0, 200.0], dtype=np.float64)

param = Fitting(x, y)
residual = ResidualAnalysis(x, y, param)
Bootstraping(x, y, residual, param)


