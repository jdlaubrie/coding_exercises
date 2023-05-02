# python3
import numpy as np
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, chi2

class Fitting(object):

    def __init__(self, x, y, param0, param_bool,
            sigma2 = None,
            regression = "linear",
            analysis = "univariate",
            maximum_iterations=50,
            marquardt_tolerance=5.0e-2,
            funcs = None):

        self.regression = regression
        self.analysis = analysis
        self.maximum_iterations = maximum_iterations
        self.marquardt_tolerance = marquardt_tolerance

        if funcs == None and regression == "nonlinear":
            print("As the function is void, a Fung model is set.")
            funcs = self.FungOrthotropic

        param, covariance = self.CurveFit(x, y, param0, param_bool, funcs, 
                                                                        sigma2)

        self.DeterminationCoefficient(x, y, param, funcs)
        self.CorrelationAssessment(param, param_bool, covariance)
        residual = self.ResidualAnalysis(x, y, param, funcs)
        #if self.analysis == "univariate":
        #    self.BootstrapingUni(x, y, residual, param0, param, params_bool,
        #                                                                 funcs)
        #elif self.analysis == "multivariate":
        #    self.BootstrapingMulti(x, y, residual, param0, param, params_bool,
        #                                                                 funcs)
        self.MakePlots(x, y, param, funcs)

#==============================================================================#
    def CurveFit(self, x, y, param0, param_bool, funcs, sigma2=None):

        if sigma2 == None:
            compute_variance = True
            ntrial = 2
            if len(y.shape) == 1:
                sigma2 = 1.0
            else:
                sigma2 = np.eye(y.shape[1],dtype=np.float64)
        else:
            compute_variance = False
            ntrial = 1

        for i in range(ntrial):
            print("===========================================================")
            print("Trial number {} to get parameters".format(i))
            if self.regression == "linear":
                print("Performing linear regression of the data.")
                if len(y.shape) == 1:
                    print("The linear regression is univariate.")
                    param, cov, chi_square = self.LinearRegressionUni(x, y, 
                                                                        sigma2)
                else:
                    print("The linear regression is multivariate.")
                    param, cov, chi_square = self.LinearRegressionMulti(x, y, 
                                                                        sigma2)
            elif self.regression == "nonlinear":
                print("Performing non-linear regression of the data.")
                param, cov, chi_square = self.NonLinearRegression(x, y, sigma2, 
                                                     param0, param_bool, funcs)
            else:
                raise ValueError("Regression type not understood.")

            print("The parameters are: \n{}".format(param))
            #print("The covariance is: \n{}".format(cov))
            print("chi-square is: {}".format(chi_square))

            if compute_variance:
                sigma2 = self.GetEstimatedVariance(x, y, param, funcs)
                print("The estimated variance is: \n{}".format(sigma2))
                compute_variance = False

        return param, cov

#==============================================================================#
    """Linear functions for the univariate and multivariate cases"""

    def LinearFunctionUni(self, x, b0, b1, b2=None):

        if len(x.shape) == 1:
            y = b0 + b1*x
        else:
            y = b0 + b1*x[:,0] + b2*x[:,1]

        return y

    def LinearFunctionMulti(self, x, b10, b11, b12, b20, b21, b22):

        y1 = b10 + b11*x[:,0] + b12*x[:,1]
        y2 = b20 + b21*x[:,0] + b22*x[:,1]

        return y1, y2

#==============================================================================#
    """Nonlinear functions for uniaxial tests"""

    def YeohUniaxial(self, x, c10, c20, c30):

        dsda = np.zeros((x.shape[0],3),dtype=np.float64)
        I1 = x**2 + 2./x

        stress = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x**2 - 1.0/x)

        dsda[:,0] = 2.0*(x**2 - 1.0/x)
        dsda[:,1] = 4.0*(I1-3.0)*(x**2 - 1.0/x)
        dsda[:,2] = 6.0*((I1-3.0)**2)*(x**2 - 1.0/x)

        return stress, dsda

#==============================================================================#
    """Nonlinear functions for biaxial tests"""

    def YeohBiaxial(self, x, c10, c20, c30):

        x1 = x[:,0]
        x2 = x[:,1]

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        I1 = x1**2 + x2**2 + 1.0/(x1*x2)
        stress[:,0] = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x1**2 - (x1*x2)**(-2))
        stress[:,1] = (2.0*c10 + 4.0*c20*(I1-3.0) + \
                6.0*c30*((I1-3.0)**2))*(x2**2 - (x1*x2)**(-2))

        dsda = np.zeros((x.shape[0],2,3),dtype=np.float64)
        dsda[:,0,0] = 2.0*(x1**2 - (x1*x2)**(-2))
        dsda[:,0,1] = 4.0*(I1-3.0)*(x1**2 - (x1*x2)**(-2))
        dsda[:,0,2] = 6.0*((I1-3.0)**2)*(x1**2 - (x1*x2)**(-2))
        dsda[:,1,0] = 2.0*(x2**2 - (x1*x2)**(-2))
        dsda[:,1,1] = 4.0*(I1-3.0)*(x2**2 - (x1*x2)**(-2))
        dsda[:,1,2] = 6.0*((I1-3.0)**2)*(x2**2 - (x1*x2)**(-2))

        return stress, dsda

    def FungOrthotropic(self, x, c, b11, b22, b33, b12, b13, b23):
        """Fung function for orthotropic stress"""

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = 1.0/(x1*x2)

        e1 = 0.5*(x1**2-1.0)
        e2 = 0.5*(x2**2-1.0)
        e3 = 0.5*(x3**2-1.0)

        Q = b11*e1**2 + b22*e2**2 + b33*e3**2 + \
            2.0*b12*e1*e2 + 2.0*b13*e1*e3 + 2.0*b23*e2*e3 #+ \
            #b44*e4**2 + b55*e5**2 + b66*e6**2

        sum1 = (b11*e1 + b12*e2 + b13*e3)*x1**2
        sum2 = (b12*e1 + b22*e2 + b23*e3)*x2**2
        sum3 = (b13*e1 + b23*e2 + b33*e3)*x3**2

        stress = np.zeros((x.shape[0],2),dtype=np.float64)
        stress[:,0] = c*np.exp(Q)*(sum1 - sum3)
        stress[:,1] = c*np.exp(Q)*(sum2 - sum3)

        dsda = np.zeros((x.shape[0],2,7),dtype=np.float64)
        dsda[:,0,0] = np.exp(Q)*(sum1 - sum3)
        dsda[:,0,1] = c*np.exp(Q)*(e1*x1**2 + (sum1-sum3)*e1**2)
        dsda[:,0,2] = c*np.exp(Q)*(sum1-sum3)*e2**2
        dsda[:,0,3] = c*np.exp(Q)*(-e3*x3**2 + (sum1-sum3)*e3**2)
        dsda[:,0,4] = c*np.exp(Q)*(e2*x1**2 + 2.0*(sum1-sum3)*e1*e2)
        dsda[:,0,5] = c*np.exp(Q)*(e3*x1**2 - e1*x3**2 + 2.0*(sum1-sum3)*e1*e3)
        dsda[:,0,6] = c*np.exp(Q)*(-e2*x3**2 + 2.0*(sum1-sum3)*e2*e3)
        dsda[:,1,0] = np.exp(Q)*(sum2 - sum3)
        dsda[:,1,1] = c*np.exp(Q)*(sum2-sum3)*e1**2
        dsda[:,1,2] = c*np.exp(Q)*(e2*x2**2 + (sum2-sum3)*e2**2)
        dsda[:,1,3] = c*np.exp(Q)*(-e3*x3**2 + (sum2-sum3)*e3**2)
        dsda[:,1,4] = c*np.exp(Q)*(e1*x2**2 + 2.0*(sum2-sum3)*e1*e2)
        dsda[:,1,5] = c*np.exp(Q)*(-e1*x3**2 + 2.0*(sum2-sum3)*e1*e3)
        dsda[:,1,6] = c*np.exp(Q)*(e3*x2**2 - e2*x3**2 + 2.0*(sum2-sum3)*e2*e3)

        return stress, dsda

#==============================================================================#
    def LinearRegressionUni(self, x, y, sigma):
        """Linear regression for the case univariate (nfun=1) and
        multivariable (nvar>=1)"""

        # redefine x by adding a row of ones, [1 x]^T
        ndata = x.shape[0]
        x = np.hstack((np.ones((ndata,1),dtype=np.float64),x))

        for i in range(2):
            # compute the maximum likelihood paramters
            sumy = np.dot(np.transpose(x),y/sigma**2)
            sumx2 = np.dot(np.transpose(x)/sigma**2,x)
            sumx2_inv = np.linalg.inv(sumx2)
            param = np.dot(sumy,sumx2_inv)
            nparam = param.shape[0]

            # measure the error of the fitting
            error = y - np.dot(x,param)

            chi_square = np.dot(error/sigma,error/sigma)

            # compute standard deviation (sigma) if it is one initially
            if np.all(np.equal(sigma,1.0)):
                sigma = np.sqrt(np.dot(error,error)/(ndata-nparam))
            else:
                break

        # gradient of the function f in the direction of the parameters b
        dfdb = np.transpose(x)
        covariance = np.dot(np.transpose(dfdb),dfdb)
        correlation = np.zeros((nparam,nparam),dtype=np.float64)
        for i in range(nparam):
            for j in range(nparam):
                correlation[i,j] = covariance[i,j]/\
                               np.sqrt(covariance[i,i]*covariance[j,j])

        return param, correlation, chi_square

#==============================================================================#
    def LinearRegressionMulti(self,x, y, sigma2):
        """Linear regression for the case multivariate (nfun>1) and
        multivariable (nvar>=1)"""

        # redefine x by adding a row of ones, [1 x]^T
        ndata = y.shape[0]
        x = np.hstack((np.ones((ndata,1),dtype=np.float64),x))

        nvar = x.shape[1]
        nfun = y.shape[1]

        dbdb = np.zeros((nvar,nfun,nvar,nfun),dtype=np.float64)
        #dfdb = np.zeros((nvar,nfun,nvar,nfun),dtype=np.float64)
        for i in range(nvar):
            for j in range(nfun):
                dbdb[i,j,i,j] = 1.0

        for count in range(2):
            sigma2_inv = np.linalg.inv(sigma2)
            dfdb = np.einsum('ij,jklm->iklm',x,dbdb)
            # compute the maximum likelihood paramters
            sum_ys = np.einsum('ij,jk->ik',y,sigma2_inv)
            sum_ysx = np.einsum('ij,ijkl->kl',sum_ys,dfdb)   # first sum
            sum_sx = np.einsum('jk,iklm->ijlm',sigma2_inv,dfdb)
            sum_xsx = np.einsum('ij,iklm->jklm',x,sum_sx)    # matrix next to beta

            # flatten the beta matrix (parameters)
            sum_ysx1 = np.zeros((nvar*nfun),dtype=np.float64)
            sum_xsx1 = np.zeros((nvar*nfun,nvar*nfun),dtype=np.float64)
            for i in range(nvar):
                for j in range(nfun):
                    sum_ysx1[j*nvar+i] = sum_ysx[i,j]
                    for k in range(nvar):
                        for l in range(nfun):
                            sum_xsx1[j*nvar+i,l*nvar+k] = sum_xsx[i,j,k,l]

            # solve the linear system
            sum_xsx1_inv = np.linalg.inv(sum_xsx1)
            param = np.dot(sum_ysx1,sum_xsx1_inv)
            nparam = param.shape[0]

            # measure the error of the fitting
            param1 = np.zeros((nvar,nfun),dtype=np.float64)
            for i in range(nvar):
                for j in range(nfun):
                    param1[i,j] = param[j*nvar+i]
            error = y - np.dot(x,param1)

            chi_square = np.einsum("ij,jk,ik",error,sigma2_inv,error)

            # compute an estimate of covariance
            if count == 0:
                sigma2 = np.einsum("ij,ik->jk",error,error)
                sigma2 /= (ndata-nparam)

        # compute covariance
        dfdb_red = np.zeros((ndata*nfun,nvar*nfun),dtype=np.float64)
        for i in range(ndata):
            for j in range(nfun):
                for k in range(nvar):
                    for l in range(nfun):
                        dfdb_red[i*nfun+j,l*nvar+k] = dfdb[i,j,k,l]

        covariance = np.dot(np.transpose(dfdb_red),dfdb_red)
        covariance = np.linalg.inv(covariance)

        return param, covariance, chi_square

#==============================================================================#
    def NonLinearRegression(self, x, y, sigma2, param, param_bool, funcs):
        """mrqmin: Levenberg-Marquardt method for fitting nonlinear
        models.
        Basicaly this method handles the speed of convergence in the 
        linearization of the least square method by a parameter alamda.
        Where beta is the gradient of chi-square and alpha is the hessian, and
        solving the system alpha*da=beta."""

        # x:           variables. [ndata,P]
        # y:           variates. [ndata,Q]
        # sigma2:      estimated variance. scalar or square matrix [Q,Q].
        # param:       coefficients of a nonlinear function. [nparam]
        # param_bool:  boolean array to select the coeff to be fitted. [nparam]
        # funcs:       the nonlinear function to fit, funcs(x,*param)

        tolerance = self.marquardt_tolerance

        nparam = param.shape[0]    # number of coefficients in the function

        # set the number of coefficients to fit
        mfit = np.sum(param_bool)  # counter for the number of coeffcients to fit
        #for j in range(nparam):
        #    if param_bool[j]: mfit += 1  # verify if ia[j] is True

        # compute the inverse of variance matrix
        sigma2_inv = np.linalg.inv(sigma2)

        # first computation of the first and second variation of chi-squared
        alamda = 0.001
        alpha, beta, ochisq = self.ChiSquareVariations(x, y, sigma2_inv,
                                                    param, param_bool, funcs)

        # define a trial parameter vector
        param_try = np.zeros((nparam),dtype=np.float64)
        param_try[param_bool] = param[param_bool]

        # initialize the residual norm (beta), this should get to zero
        norm_beta = np.linalg.norm(beta)
        if np.isclose(norm_beta,0.0):
            norm_beta = 1e-14
        norm_residual = np.abs(np.linalg.norm(beta)/norm_beta)

        # loop of iteration for Levenberg-Marquardt method
        iteration = 0
        while norm_residual > tolerance:

            # define the linear system to solve. alpha*da=beta
            # alter alpha by augmenting diagonal elements
            alpha_lm = alpha + alamda*np.diag(np.diag(alpha))
            # invert hessian matrix and get the solution of covar*beta=da
            alpha_lm_inv = np.linalg.inv(alpha_lm)
            da = np.dot(alpha_lm_inv,beta)

            # update the coefficients
            param_try[param_bool] = param[param_bool] + da

            # compute the residual for this iteration
            abs_residual = np.linalg.norm(beta)
            norm_residual = np.abs(np.linalg.norm(beta)/norm_beta)

            print("Iteration {}.".format(iteration) +\
                  " Lambda {0:>10.5g}.".format(alamda) +\
                  " Residual (abs) {0:>16.7g}.".format(abs_residual) +\
                  " Residual (rel) {0:>16.7g}.".format(norm_residual))

            # verify the succes of the trial
            alpha_try, beta_try, chisq = self.ChiSquareVariations(x, y,
                                    sigma2_inv, param_try, param_bool, funcs)
            if chisq < ochisq:  # the trial is better, accept the new solution
                alamda = 0.1*alamda
                ochisq = chisq
                alpha = alpha_try
                beta = beta_try
                param[param_bool] = param_try[param_bool]
            else:  # the trial is worse, increase alamda and return
                alamda = 10.0*alamda

            # break if the absolute residual is small enough
            if np.abs(abs_residual) < tolerance:
                break

            # break if the coefficients variation is small enough
            if np.linalg.norm(da) < tolerance:
                break

            # update iteration number
            iteration += 1

            if np.isnan(norm_residual) or norm_residual>1e6:
                raise RuntimeError("The solution diverges.")

            if iteration == self.maximum_iterations-1:
                raise RuntimeError("Maximum number of iterations reached.")

        # prepare the covariance matrix for output
        alpha_inv = np.linalg.inv(alpha)
        covariance = self.CovarianceMatrix(alpha_inv, param_bool)

        return param, covariance, ochisq

#==============================================================================#
    def ChiSquareVariations(self, x, y, sigma2_inv, param, param_bool, funcs):
        """mrqcof: used by mrqmin to evaluate the linearized fitting matrix
        alpha, and the vector beta, and calculate chisq"""

        # x (IN):          set of datapoints. [ndata]
        # y (IN):          set of datapoints. [ndata]
        # sigma2 (IN):     estimated variance.
        # param (INOUT):   coefficients of a nonlinear function. [ma]
        # param_bool (IN): boolean array to select the coeff to be fitted. [ma]
        # alpha (OUT):     curvature matrix (hessian). [ma,ma]
        # beta (OUT):      vector (first derivatives). [ma]
        # chisq (OUT):     chi square, the functional to minimize
        # funcs (IN):      the nonlinear function to fit, funcs(x,a,yfit,dyda,ma)

        if len(y.shape) == 1:
            f_model, dfdb = funcs(x,*param)
            if len(dfdb.shape) == 1:
                dfdb = np.reshape(dfdb,(dfdb.shape[0],1))

            error = y - f_model
            
            hessian0 = np.einsum("ij,ik->jk",dfdb,dfdb)
            hessian0 = sigma2_inv*hessian0
            hessian = hessian0[param_bool,:][:,param_bool]

            gradient0 = np.einsum("i,ik->k",error,dfdb)
            gradient0 = sigma2_inv*gradient0
            gradient = gradient0[param_bool]

            chi_square = np.dot(error,error)*sigma2_inv

        else:
            f_model, dfdb = funcs(x,*param)

            error = y - f_model

            hessian0 = np.einsum("ijl,jk->ikl",dfdb,sigma2_inv)
            hessian0 = np.einsum("ijk,ijl->kl",hessian0,dfdb)
            hessian = hessian0[param_bool,:][:,param_bool]

            gradient0 = np.einsum("ij,jk->ik",error,sigma2_inv)
            gradient0 = np.einsum("ij,ijk->k",gradient0,dfdb)
            gradient = gradient0[param_bool]

            chi_square = np.einsum("ij,jk,ik",error,sigma2_inv,error)
        
        return hessian, gradient, chi_square

#==============================================================================#
    def CovarianceMatrix(self, alpha_inv, param_bool):
        """covsrt: spread the covariances back into the full nparam*nparam
        covariance matrix. Covariance computed for the parameters."""

        # covariance (OUT): covariance matrix. covar[ma,ma]
        # param_bool (IN):  boolean array with the coeff to be fitted. [nparam]

        nparam = param_bool.shape[0]     # number of coeff in the function
        covariance = np.zeros((nparam,nparam),dtype=np.float64)

        k = 0
        for i in range(nparam):
            if param_bool[i]:
                l = 0
                for j in range(nparam):
                    if param_bool[j]:
                        covariance[i,j] = alpha_inv[k,l]
                        l += 1
                k += 1

        return covariance

#==============================================================================#
    def GetEstimatedVariance(self, x, y, param, funcs):

        nparam = param.shape[0]     # number of coefficients in the function
        ndata = y.shape[0]     # number of coefficients in the function

        if len(y.shape) == 1:
            f_model, _ = funcs(x,*param)
            error = y - f_model

            sigma2 = np.dot(error,error)
            sigma2 /= (ndata-nparam)

        else:
            f_model, _ = funcs(x,*param)
            error = y - f_model

            sigma2 = np.einsum("ij,ik->jk",error,error)
            sigma2 /= (ndata-nparam)

        return sigma2

#==============================================================================#
    def CorrelationAssessment(self, params, param_bool, covariance):

        nparam = params.shape[0]

        mfit = 0  # counter for the number of coefficients to fit
        for j in range(nparam):
            if param_bool[j]: mfit += 1

        covariance = covariance[param_bool,:][:,param_bool]

        correlation = np.zeros((mfit,mfit),dtype=np.float64)
        for i in range(mfit):
            for j in range(mfit):
                correlation[i,j] = covariance[i,j]/\
                           np.sqrt(covariance[i,i]*covariance[j,j])

        determinant = np.linalg.det(correlation)
        if determinant < 1.0e-4:
            is_overparameter = True
        else:
            is_overparameter = False

        print("This is the correlation: \n{}".format(correlation))
        print("Over-parameterization: {}, ".format(is_overparameter) + \
              "det(R) = {}".format(determinant) )

#==============================================================================#
    def DeterminationCoefficient(self, x, y, params, funcs):

        from sklearn.metrics import r2_score
        y_pred, _ = funcs(x, *params)
        determination = r2_score(y, y_pred, multioutput='raw_values')
        print("Determination Coefficient R2: {}".format(determination))

#==============================================================================#
    def ResidualAnalysis(self, x, y, params, funcs):

        error = y - funcs(x, *params)[0]

        fig, ax = plt.subplots(2,3, figsize=plt.figaspect(0.5))

        # histogram for error 1
        ax[0,0].hist(error[:,0], bins=6)
        ax[0,0].set_xlabel('error 1',fontsize=11)
        ax[0,0].set_ylabel('quantity',fontsize=11)

        # histogram for error 1
        ax[1,0].hist(error[:,1], bins=6)
        ax[1,0].set_xlabel('error 2',fontsize=11)
        ax[1,0].set_ylabel('quantity',fontsize=11)

        # scatter stretch against error
        ax[0,1].scatter(x[:,0], error[:,0])
        ax[0,1].scatter(x[:,1], error[:,1])
        ax[0,1].set_xlabel('stretch',fontsize=11)
        ax[0,1].set_ylabel('error',fontsize=11)

        # plot error 1 against error 2
        ax[1,1].plot([min(error[:,0]),max(error[:,0])],
                     [min(error[:,0]),max(error[:,0])],
                     color='red')
        ax[1,1].scatter(error[:,0], error[:,1])
        ax[1,1].set_xlabel('error 1',fontsize=11)
        ax[1,1].set_ylabel('error 2',fontsize=11)

        # QQplot for error 1
        sm.graphics.qqplot(error[:,0], line='s', ax=ax[0,2])

        # QQplot for error 2
        sm.graphics.qqplot(error[:,1], line='s', ax=ax[1,2])

        fig.tight_layout()
        plt.show
        #file output
        FIGURENAME = 'error.pdf'
        plt.savefig(FIGURENAME)
        #close graphical tools
        plt.close('all')

        return error

#==============================================================================#
    def BootstrapingUni(self, x, y, residual, param0, param, param_bool, funcs, 
                                                             around_func=True):

        ndata = x.shape[0]
        nparam = param0.shape[0]
        conf = 0.95
        nboot = 100

        # creates bootstrap residuals
        boot_res = np.random.choice(residual,(nboot,ndata))
        boot_y = np.zeros((nboot,ndata),dtype=np.float64)
        boot_opt = np.zeros((nboot,nparam),dtype=np.float64)
        if around_func:
            for i in range(nboot):
                boot_y[i,:] = boot_res[i,:] + funcs(x,*param)
                boot_opt[i,:], _ = curve_fit(funcs, x, boot_y[i,:],
                                             p0=param0)
        else:
            for i in range(nboot):
                boot_y[i,:] = boot_res[i,:] + y
                boot_opt[i,:], _ = curve_fit(funcs, x, boot_y[i,:],
                                             p0=param0)

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
        ax.plot(x, funcs(x, *param), label='fit')
        ax.legend(loc='upper left')

        fig.tight_layout()
        plt.show
        FIGURENAME = 'bootstrap_fit.png'
        plt.savefig(FIGURENAME)
        plt.close('all')

#==============================================================================#
    def BootstrapingMulti(self, x, y, residual, param0, param, param_bool, funcs, 
                                                             around_func=True):

        ndata = x.shape[0]
        nfunc = y.shape[1]
        nparam = param0.shape[0]
        mfit = np.sum(param_bool)
        conf = 0.95
        nboot = 100

        # creates bootstrap residuals
        boot_res = np.zeros((nboot,ndata,nfunc),dtype=np.float64)
        for i in range(nfunc):
            boot_res[:,:,i] = np.random.choice(residual[:,i],(nboot,ndata))
        boot_y = np.zeros((nboot,ndata,nfunc),dtype=np.float64)
        boot_opt0 = np.zeros((nboot,nparam),dtype=np.float64)
        if around_func:
            for i in range(nboot):
                boot_y[i,:,:] = boot_res[i,:,:] + funcs(x, *param)[0]
                boot_opt0[i,:], _ = self.CurveFit(x, boot_y[i,:,:], param0,
                                                     param_bool, funcs)
        else:
            for i in range(nboot):
                boot_y[i,:,:] = boot_res[i,:,:] + y
                boot_opt0[i,:], _ = self.CurveFit(x, boot_y[i,:,:], param0,
                                                     param_bool, funcs)

        boot_opt = boot_opt0[:,param_bool]
        boot_mean = np.mean(boot_opt,axis=0)
        boot_std = np.std(boot_opt, axis=0, ddof=1)
        boot_cov = np.cov(boot_opt.transpose())

        print("\nBootstrap mean = {}".format(boot_mean))
        print("Bootstrap std = {}".format(boot_std))

        z = np.linspace(boot_mean-3.*boot_std,boot_mean+3.*boot_std,100)

        # assumed normal distribution for bootstrap data
        dist0 = norm(loc=boot_mean, scale=boot_std)
        interval0 = dist0.interval(conf)
        p_value0 = dist0.cdf(param[param_bool])

        # boot_mean + norm(conf)*boot_std
        print("\nSeparate confidence interval ({}) ".format(conf*100.) +\
              "= \n{}".format(interval0))
        print("The p-values are = {}".format(p_value0))

        # multivariate test
        dist3 = chi2(mfit)
        diff = boot_mean - param[param_bool]
        boot_cov_inv = np.linalg.inv(boot_cov)
        z_square = np.einsum('i,ij,j',diff,boot_cov_inv,diff)
        print("\nZ-squared (multivariate test) = {}".format(z_square))
        print("chi-squared at {} ".format(100.*conf/2.) +\
          "= {}".format(dist3.ppf(conf/2.)) )
        #print("P-value of test = {}".format(dist3.cdf(z_square)))

#------------------------------------------------------------------------------#
        factor = nboot*boot_std/np.sqrt(ndata)
        fig, ax = plt.subplots(2,mfit, figsize=plt.figaspect(0.5))

        for i in range(mfit):
            # data distribution
            ax[0,i].hist(boot_opt[:,i], bins=15)
            ax[0,i].plot(z[:,i], factor[i]*dist0.pdf(z)[:,i])
            # QQplot for error 1
            sm.graphics.qqplot(boot_opt[:,i], line='s', ax=ax[1,i])

        fig.tight_layout()
        plt.show
        FIGURENAME = 'bootstrap.png'
        plt.savefig(FIGURENAME)
        plt.close('all')

#------------------------------------------------------------------------------#
        fig, ax = plt.subplots()

        # data distribution
        for i in range(nboot):
            ax.scatter(x, boot_y[i,:,:])
        ax.plot(x, y, marker='*', color='red', label='exp')
        ax.plot(x, funcs(x, *param)[0], label='fit')
        ax.legend(loc='upper left')

        fig.tight_layout()
        plt.show
        FIGURENAME = 'bootstrap_fit.png'
        plt.savefig(FIGURENAME)
        plt.close('all')

#==============================================================================#
    def MakePlots(self, x, y, params, funcs):
        """This is a function to make a general plot of the fitting"""

        fig, ax = plt.subplots(constrained_layout=True)

        # plot curve and data
        ax.plot(x, funcs(x, *params)[0], 'b-', label='fitting')
        ax.plot(x, y, 'o', label='data')

        # make the graphic nicer
        ax.set_xlabel('stretch',fontsize=14)
        ax.set_ylabel('stress',fontsize=14)
        ax.set_ylim(bottom=0,top=250)
        ax.set_xlim(left=1.0,right=1.17)
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

        if len(y.shape) != 1:
            from mpl_toolkits.mplot3d import Axes3D

            X = np.linspace(1.0,1.17,7)
            Y = np.linspace(1.0,1.1,7)
            X, Y = np.meshgrid(X,Y)
            Z1 = np.zeros((X.shape[0],X.shape[1]),dtype=np.float64)
            Z2 = np.zeros((X.shape[0],X.shape[1]),dtype=np.float64)
            for i in range(X.shape[0]):
                XY_data = np.column_stack((X[i,:],Y[i,:]))
                Z1[i,:] = funcs(XY_data,*params)[0][:,0]
                Z2[i,:] = funcs(XY_data,*params)[0][:,1]

            # first plot
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')

            ax.plot_wireframe(X, Y, Z1)
            ax.plot(x[:,0],x[:,1],y[:,0], 'b-')

            ax.set_xlabel('stretch 1',fontsize=11)
            ax.set_ylabel('stretch 2',fontsize=11)
            ax.set_zlabel('stress',fontsize=11)

            # second plot
            ax = fig.add_subplot(1, 2, 2, projection='3d')

            ax.plot_wireframe(X, Y, Z2)
            ax.plot(x[:,0],x[:,1],y[:,1], 'r-')

            ax.set_xlabel('stretch 1',fontsize=11)
            ax.set_ylabel('stretch 2',fontsize=11)
            ax.set_zlabel('stress',fontsize=11)

            plt.show

            FIGURENAME = 'fitting_3d.pdf'
            plt.savefig(FIGURENAME)

            plt.close('all')

#==============================================================================#

x = np.array([[1.041, 1.059, 1.098, 1.119, 1.131, 1.140, 1.148, 1.153, 1.157,
        1.162, 1.166],[1.020, 1.029, 1.046, 1.056, 1.063, 1.068, 1.071, 1.075,
        1.077, 1.080, 1.081]], dtype=np.float64)
y = np.array([[10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0,
        140.0, 160.0, 180.0, 200.0],[10.0, 20.0, 40.0, 60.0,
        80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]], 
        dtype=np.float64)

# insert always x and y with the ndata in the first index 
x = np.transpose(x)    # [ndata,p] p is the number of variables
y = np.transpose(y)    # [ndata,q] q is the number of variates

#y *= x
y[:,0] *= x[:,0]
y[:,1] *= x[:,1]

params = np.array([10.0,1.0,1.0,0.0,0.0,0.0,0.0],dtype=np.float64)
params_bool = np.array([True,True,True,False,True,False,False],dtype=bool)

fitting = Fitting(x, y, params, params_bool,
                    regression="nonlinear",
                    analysis="multivariate")

