# Numerical library
import numpy as np

def func_eval(time):
    """
    Function with the aim to get the values of the function and derivative at x.
    """
    func = 120.0*np.sin(np.pi*time/0.3) - 7.6
    df = 120.0*(np.pi/0.3)*np.cos(np.pi*time/0.3)
    return func,df

def newton_raphson(t1,t2,tol):
    """
    Using the Newton-Raphson method, find the root of a function known to lie close to x. 
    The root rtnewt will be refined until its accuracy is known within +/- xacc. funcd
    is a user-supplied subroutine that returns both the function value and the first derivative
    of the function at the point x.
    """
    # Set to maximum number of iterations.
    JMAX=20
    # Initial guess.
    rtnewt=0.5*(t1+t2)
    #res,_ = func_eval(time=time)
    for j in range(JMAX):
        func,df = func_eval(rtnewt)
        dx = func/df
        rtnewt -= dx
        # Error
        if (t1-rtnewt)*(rtnewt-t2)<0.0:
            print("rtnewt jumped out of brackets")
            return rtnewt
        # Convergence.
        if np.absolute(dx)<tol: #np.absolute(func/res)<tolerance:
            return rtnewt
    print('newton_raphson exceeded maximum iterations')
    return rtnewt

# Test of the Solution method with derivatives from Ridders method
t1 = 0.2
t2 = 0.35
tol = 1.0e-5
solution = newton_raphson(t1,t2,tol)
print(solution)

#--------------------------------------------------------------------------
#          curves visualization. graphics
#--------------------------------------------------------------------------
import matplotlib.pyplot as plt
time = np.linspace(0.0,0.3,num=15)
cte = 7.6*np.ones(15)
fig, ax = plt.subplots(constrained_layout=True)

# plot curve and data
ax.plot(time, 120.0*np.sin(np.pi*time/0.3), 'b-')
ax.plot(time, cte, 'r-')

# make the graphic nicer
ax.set_xlabel('time [s]',fontsize=14)
ax.set_ylabel('pressure [mmHg]',fontsize=14)
for label in (ax.get_xticklabels()+ax.get_yticklabels()):
    label.set_fontsize(14)
ax.grid(True)

plt.show()
#close graphical tools
plt.close('all')

