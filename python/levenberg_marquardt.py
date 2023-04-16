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

    # x (IN):         set of datapoints. [ndata]
    # y (IN):         set of datapoints. [ndata]
    # sig (IN):       individual standard deviation. [ndata]
    # a (INOUT):      coefficients of a nonlinear function. [ma]
    # ia (IN):        boolean array to select the coeff to be fitted. [ma]
    # alpha (OUT):    curvature matrix (hessian). [ma,ma]
    # beta (OUT):     vector (first derivatives). [ma]
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

def LevenbergMarquardt(x, y, sigma, param, param_bool, funcs):
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

    nparam = param.shape[0]    # number of coefficients in the function

    covariance = np.zeros((nparam,nparam),dtype=np.float64)
    alpha = np.zeros((nparam,nparam),dtype=np.float64)
    beta = np.zeros((nparam),dtype=np.float64)
    chi_square = 0.0
    alamda = -1.0

    # set the number of coefficients to fit
    mfit = 0  # counter for the number of coeffcients to fit
    for j in range(nparam):
        if param_bool[j]: mfit += 1  # verify if ia[j] is True

    maximum_iteration = 15
    residual = 1.0
    for i in range(maximum_iteration):

        param_try = np.zeros((nparam),dtype=np.float64)

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
        covariance[:mfit,:mfit] = np.linalg.inv(covariance[:mfit,:mfit])
        da = np.dot(covariance[:mfit,:mfit],da)

        # once converged, evaluate covariance matrix
        if np.isclose(alamda,0.0):
            # spread out alpha to its full size
            covariance = CovarianceMatrix(covariance, param_bool)
            alpha = CovarianceMatrix(alpha, param_bool)
            break

        # update the coefficients
        j = 0  # counter for the number of coefficients to fit
        for l in range(nparam):
            if param_bool[l]:
                param_try[l] = param[l] + da[j]
                j += 1

        if i != 0:
            residual = np.abs(np.linalg.norm(da))/np.linalg.norm(param_try)

        print("Iteration {}.".format(i) +\
              " Lambda {0:>16.7g}.".format(alamda) +\
              " Residual {0:>16.7g}.".format(residual))

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

        if residual < 1.0e-2:
            alamda = 1.0e-14

        if i == maximum_iteration-1:
            print("Maximum number of iterations reached.")

    return param, covariance, ochisq

#==============================================================================#
x = np.array([1.01, 1.020, 1.024, 1.029, 1.034, 1.040, 1.046, 1.053, 1.061,
        1.068],dtype=np.float64)
y = np.array([32.748, 72.515, 114.620, 152.047, 194.152, 233.918, 273.684,
        315.789, 355.556, 392.982],dtype=np.float64)
y *= x
sigma = np.ones((x.shape[0]),dtype=np.float64)

params = np.array([1.0,1.0,1.0],dtype=np.float64)
params_bool = np.array([True,False,False],dtype=bool)
nparams = params.shape[0]

def funcs(x, c10, c20, c30):
    dsda = np.zeros((x.shape[0],3),dtype=np.float64)
    I1 = 2.*x**2 + 1./x**4
    stress = 2.*c10*(x**2 - 1./x**4) \
           + 4.*c20*(I1-3.)*(x**2 - 1./x**4) \
           + 6.*c30*((I1-3.)**2)*(x**2 - 1./x**4)
    dsda[:,0] = 2.*(x**2 - 1.0/x**4)
    dsda[:,1] = 4.*(I1-3.)*(x**2 - 1./x**4)
    dsda[:,2] = 6.*((I1-3.)**2)*(x**2 - 1./x**4)
    return stress, dsda

params, covariance, chi_square = LevenbergMarquardt(x, y, sigma, params, 
            params_bool, funcs)

print(params)
print(covariance)
print(chi_square)

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

