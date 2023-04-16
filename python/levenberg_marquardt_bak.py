# python3
import numpy as np
import sys

def CovarianceMatrix(covar,ia):
    """covsrt: spread the covariances back into the full ma*ma covariance
    matrix"""

    # INOUT arguments
    #INTEGER ma,mfit
    #REAL covar(ma,ma)
    #LOGICAL ia(ma)

    # covar (OUT):  covariance matrix. covar[ma,ma]
    # ia (IN):      boolean array to select the coeff to be fitted. ia[ma]
    # mfit (IN):    number of coefficients to fit

    ma = ia.shape[0]     # number of coefficients in the function

    # internal arguments
    #INTEGER i,j,k
    #REAL swap

    mfit = 0  # counter for the number of coefficients to fit
    for j in range(ma):
        if ia[j]: mfit += 1

    covar[mfit:ma,:ma] = 0.0
    covar[:ma,mfit:ma] = 0.0

    k = mfit-1
    for j in range(ma-1,-1,-1):
        if ia[j]:
            for i in range(ma):
                swap = covar[i,k]
                covar[i,k] = covar[i,j]
                covar[i,j] = swap
            for i in range(ma):
                swap = covar[k,i]
                covar[k,i] = covar[j,i]
                covar[j,i] = swap
            k -= 1
    return covar

def MarquardtCoefficients(x,y,sig,a,ia,alpha,beta,funcs):
    """mrqcof: used by mrqmin to evaluate the linearized fitting matrix alpha,
    and the vector beta, and calculate chisq"""

    # INOUT arguments
    #INTEGER ia(ma),MMAX
    #REAL sig(ndata),x(ndata),y(ndata)
    #REAL chisq
    #REAL a(ma),alpha(ma,ma),beta(ma),

    # x (IN):         set of datapoints. x[ndata]
    # y (IN):         set of datapoints. y[ndata]
    # sig (IN):       individual standard deviation. sig[ndata]
    # a (INOUT):      coefficients of a nonlinear function. a[ma]
    # ia (IN):        boolean array to select the coeff to be fitted. ia[ma]
    # alpha (OUT):    curvature matrix (hessian). alpha[ma,ma]
    # beta (OUT):     vector (first derivatives). beta[ma]
    # chisq (OUT):    chi square, the functional to minimize
    # funcs (IN):     the nonlinear function to fit, funcs(x,a,yfit,dyda,ma)

    ma = a.shape[0]     # number of coefficients in the function

    # internal arguments
    #INTEGER mfit,i,j,k,l,m
    #REAL dy[ndata],sig2i[ndata],wt[ndata],ymod[ndata],dyda(ndata,ma)

    mfit = 0  # counter for the number of coefficients to fit
    for j in range(ma):
        if ia[j]: mfit += 1

    # initialize alpha, beta and chisq in zero
    alpha[:mfit,:mfit] = 0.0
    beta[:mfit] = 0.0

    ymod, dyda = funcs(x,*a)
    if len(dyda.shape) == 1:
        dyda = np.reshape(dyda,(dyda.shape[0],1))

    sig2i = 1.0/(sig*sig)
    dy = y - ymod
    j = 0
    for l in range(ma):
        if ia[l]:
            wt = dyda[:,l]*sig2i
            k = 0
            for m in range(l+1):
                if ia[m]:
                    alpha[j,k] = np.dot(wt,dyda[:,m])
                    alpha[k,j] = alpha[j,k]
                    k += 1
            beta[j] = np.dot(dy,wt)
            j += 1
    chisq = np.dot(dy*dy,sig2i)

    return alpha, beta, chisq

def LevenbergMarquardt(x, y, sigma, param, param_bool, covariance, alpha, beta, 
                    ochisq, funcs, alamda):
    """mrqmin: Levenberg-Marquardt method for the fitting of nonlinear
    functions"""

    # x (IN):           set of datapoints. [ndata]
    # y (IN):           set of datapoints. [ndata]
    # sigma (IN):       individual standard deviation. [ndata]
    # param (INOUT):    parameters of a nonlinear function. [nparam]
    # param_bool (IN):  boolean array to select the coeff to be fitted. [nparam]
    # covariance (OUT): covariance matrix. [nparam,nparam]
    # alpha (OUT):      curvature matrix. [nparam,nparam]
    # beta (OUT):       gradient vector. [nparam]
    # ochisq (OUT):     old chi square, the functional to minimize
    # funcs (IN):       the nonlinear function to fit, funcs(x,*param)
    # alamda (INOUT):   factor to handle the linearized system of equations

    ndata = x.shape[0]         # number of datapoints
    nparam = param.shape[0]    # number of coefficients in the function

    param_try = np.zeros((nparam),dtype=np.float64)
    #da = np.zeros((mfit),dtype=np.float64)

    # set the number of coefficients to fit
    mfit = 0  # counter for the number of coeffcients to fit
    for j in range(nparam):
        if param_bool[j]: mfit += 1  # verify if ia[j] is True

    if alamda < 0.0:
        alamda = 0.001
        # compute alpha and beta
        alpha, beta, ochisq = MarquardtCoefficients(x, y, sigma, param, 
                                        param_bool, alpha, beta, funcs)
        param_try = param

    # define the linear system to solve. alpha*da=beta
    # and alter alpha by augmenting diagonal elements
    covariance[:mfit,:mfit] = alpha[:mfit,:mfit]
    da = beta[:mfit]
    for j in range(mfit):
        covariance[j,j] = alpha[j,j]*(1.0 + alamda)

    # invert covariance matrix and get the solution of covar*x=da
    #covar[:mfit,:mfit],da = GaussJordan(covar[:mfit,:mfit],da) 
    covariance = np.linalg.inv(covariance)
    da = np.dot(covariance,da)

    # once converged, evaluate covariance matrix
    if np.isclose(alamda,0.0):
        # spread out alpha to its full size
        covariance = CovarianceMatrix(covariance, param_bool)
        alpha = CovarianceMatrix(alpha, param_bool)
        return param, covariance, alpha, beta, ochisq, alamda

    # update the coefficients
    j = 0  # counter for the number of coefficients to fit
    for l in range(nparam):
        if param_bool[l]:
            param_try[l] = param[l] + da[j]
            j += 1

    # verify the succes of the trial
    alpha, beta, chisq = MarquardtCoefficients(x, y, sigma, param_try, 
                                            param_bool, alpha, beta, funcs)
    if chisq < ochisq:  # success, accept the new solution
        alamda = 0.1*alamda
        ochisq = chisq
        alpha[:mfit,:mfit] = covariance[:mfit,:mfit]
        beta[:mfit] = da
        param = param_try
    else:  # failure, increase alamda and return
        alamda = 10.0*alamda

    return param, covariance, alpha, beta, ochisq, alamda

#==============================================================================#
x = np.array([1.01, 1.020, 1.024, 1.029, 1.034, 1.040, 1.046, 1.053, 1.061,
        1.068],dtype=np.float64)
y = np.array([32.748, 72.515, 114.620, 152.047, 194.152, 233.918, 273.684,
        315.789, 355.556, 392.982],dtype=np.float64)
y *= x
sigma = np.ones((x.shape[0]),dtype=np.float64)

params = np.array([20.0],dtype=np.float64)
params_bool = np.array([True],dtype=bool)
nparams = params.shape[0]

covariance = np.zeros((nparams,nparams),dtype=np.float64)
alpha = np.zeros((nparams,nparams),dtype=np.float64)
beta = np.zeros((nparams),dtype=np.float64)
chi_square = 0.0
alamda = -1.0

def funcs(x, c10):
    stress = 2.0*c10*(x**2 - 1./x)
    dsdc10 = 2.0*(x**2 - 1./x)
    return stress, dsdc10

maximum_iteration = 10
residual = 1.0
for i in range(maximum_iteration):
    print("Levenberg-Marquardt. Iteration {}".format(i) +\
          " Lambda is {0:>16.7g}.".format(alamda))
    params, covariance, alpha, beta, chi_square, alamda = LevenbergMarquardt(x, 
                            y, sigma, params, params_bool, covariance, alpha, 
                            beta, chi_square, funcs, alamda)

    if i != 0:
        residual = np.abs(np.linalg.norm(params-a_old))/np.linalg.norm(params)

    print("The residual is {0:>16.7g}.".format(residual))

    if np.isclose(alamda,0.0):
        break

    if residual < 1.0e-3:
        alamda = 1.0e-14

    a_old = params

print(params)
print(covariance)
print(alpha)

#--------------------------------------------------------------------------
#          compute determination coefficient
#--------------------------------------------------------------------------
from sklearn.metrics import r2_score
y_pred, _ = funcs(x, *params)
deter = r2_score(y, y_pred)
print("Determination Coefficient: {}".format(deter))

#--------------------------------------------------------------------------
#          curves visualization. graphics
#--------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(constrained_layout=True)

# plot curve and data
ax.plot(x, funcs(x, *params)[0], 'b-', label='Fit')
ax.plot(x, y, 'o', label='Experimental data')

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
FIGURENAME = 'fitting.pdf'
plt.savefig(FIGURENAME)
#close graphical tools
plt.close('all')

