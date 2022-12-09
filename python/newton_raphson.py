# Mathematics libraries
import numpy as np

def residual_func(time,Delta_t,Y,density,den_):
    # Elastin degradation [days]
    L_dam = 0.010
    t_dam = 40.0
    D_max = 0.5
    T_ela = 101.0*365.25
    density0 = 241.5
    # Residual function
    exponential = np.exp(-0.5*(Y/L_dam)**2 - time/t_dam)
    residual = -density + den_ + Delta_t*(-density/T_ela-(D_max/t_dam)*np.multiply(exponential,density0))
    return residual

def df_ridders(time,Delta_t,Y,density,den_):
    """
    Returns the derivative of a function func at a point x by Ridders method of polynomial
    extrapolation. The value h is input as an estimated initial stepsize; it need not be small,
    but rather should be an increment in x over which func changes substantially. An estimate
    of the error in the derivative is returned as err .
    Parameters: Stepsize is decreased by CON at each iteration. Max size of tableau is set by
    NTAB. Return when error is SAFE worse than the best so far.
    """
    h=1.0e-3
    BIG=1.0e30
    NTAB=10
    CON=1.4
    CON2=CON*CON
    SAFE=2.0
    if h==0.0:
        print('h must be nonzero in df_ridders')
        return
    a = np.zeros((NTAB,NTAB),dtype=np.float64)
    hh=h
    x_pos = density + hh
    x_neg = density - hh
    func_pos = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=x_pos,den_=den_)
    func_neg = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=x_neg,den_=den_)
    a[0,0] = (func_pos-func_neg)/(2.0*hh)
    err=BIG
    # Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of extrapolation.
    for i in range(1,NTAB):
        hh=hh/CON
        # Try new, smaller stepsize.
        x_pos = density + hh
        x_neg = density - hh
        func_pos = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=x_pos,den_=den_)
        func_neg = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=x_neg,den_=den_)
        a[0,i] = (func_pos-func_neg)/(2.0*hh)
        fac=CON2
        # Compute extrapolations of various orders, requiring no new function evaluations.
        for j in range(1,i+1):
            a[j,i] = (a[j-1,i]*fac-a[j-1,i-1])/(fac-1.0)
            fac=CON2*fac
            errt=max(abs(a[j,i]-a[j-1,i]),abs(a[j,i]-a[j-1,i-1]))
            #The error strategy is to compare each new extrapolation to one order lower, both at
            #the present stepsize and the previous one.
            #If error is decreased, save the improved answer.
            if (errt<=err):
                err=errt
                dfridr=a[j,i]
        #If higher order is worse by a significant factor SAFE, then quit early.
        if (abs(a[i,i]-a[i-1,i-1])>=SAFE*err):
            #print('Early quit in df_ridders function')
            return dfridr,err
    return dfridr,err

def func_eval(time,Delta_t,Y,density,den_):
    """
    Function with the aim to get the values of the function and derivative at x.
    """
    h=1.0e-3
    func = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=density,den_=den_)
    df,err = df_ridders(time=time,Delta_t=Delta_t,Y=Y,density=density,den_=den_)
    return func,df,err

def newton_raphson(time,Delta_t,Y,density):
    """
    Using the Newton-Raphson method, find the root of a function known to lie close to x. 
    The root rtnewt will be refined until its accuracy is known within +/- xacc. funcd
    is a user-supplied subroutine that returns both the function value and the first derivative
    of the function at the point x.
    """
    # Set to maximum number of iterations.
    JMAX=20
    res = residual_func(time=time,Delta_t=Delta_t,Y=Y,density=density,den_=density)
    # Initial guess.
    rtnewt=density
    for j in range(JMAX):
        func,df,err = func_eval(time=time,Delta_t=Delta_t,Y=Y,density=rtnewt,den_=density)
        #print(func)
        #print(df)
        dx = func/df
        #print(dx)
        rtnewt -= dx
        #print(rtnewt)
        # Convergence.
        if np.absolute(func/res)<1.e-5:
            #print('Iterations: '+str(j))
            return rtnewt
    print('newton_raphson exceeded maximum iterations')
    return rtnewt

# Test of the Solution method with derivatives from Ridders method
T_e = 101*365.25
Dm = 0.5
td = 40.0
Y=0.001
Ld=0.010
den_=241.5
dens=241.5
# time
Delta_t=10.0
time = 0.0
while time<10:
    time += Delta_t
    backeuler = newton_raphson(time=time,Delta_t=Delta_t,Y=Y,density=den_)
    func,df,err = func_eval(time=time,Delta_t=Delta_t,Y=Y,density=backeuler,den_=den_)
    density=241.5*np.exp(-time/T_e)+241.5*(Dm/td)*(T_e*td/(td-T_e))*np.exp(-0.5*(Y/Ld)**2)*\
        (np.exp(-time/T_e)-np.exp(-time/td))
    exponential = np.exp(-0.5*(Y/0.010)**2 - (time-Delta_t)/40.0)
    forweuler = dens + Delta_t*(-dens/(101.*365.25)-(0.5/40.)*np.multiply(exponential,241.5))
    dens = forweuler
    den_=backeuler
    print(time,forweuler,backeuler,density)

