# Mathematical library
import numpy as np

def func_val(time):
    Amp = 5.0
    T = np.pi
    # function values
    value = Amp*np.sin(time*np.pi/T)
    return value

def df_ridders(time):
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
    a[0,0] = (func_val(time+hh)-func_val(time-hh))/(2.0*hh)
    err=BIG
    # Successive columns in the Neville tableau will go to smaller stepsizes and higher orders of extrapolation.
    for i in range(1,NTAB):
        hh=hh/CON
        # Try new, smaller stepsize.
        a[0,i] = (func_val(time+hh)-func_val(time-hh))/(2.0*hh)
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

def func_eval(time):
    """
    Function with the aim to get the values of the function and derivative at x.
    """
    func = func_val(time=time)
    df,err = df_ridders(time=time)
    return func,df,err

# Test of the Solution method with derivatives from Ridders method
# time
final_time = 5.0
nstep = 20
func = np.zeros((nstep),dtype=np.float64)
df = np.zeros((nstep),dtype=np.float64)
err = np.zeros((nstep),dtype=np.float64)
time = np.linspace(0.0,final_time,num=nstep)
for i in range(nstep):
    func[i],df[i],err[i] = func_eval(time=time[i])

print(err)
# making some graphics
import matplotlib.pyplot as plt

fig,ax = plt.subplots()

ax.plot(time,func)
ax.plot(time,df)
ax.set_xlabel('time [s]',fontsize=11)
ax.grid(True)

fig.tight_layout()
plt.show()
plt.close('all')

