import numpy as np

def func_eval(time):

    func = 120.0*np.sin(np.pi*time/0.3) - 7.6
    df = 120.0*(np.pi/0.3)*np.cos(np.pi*time/0.3)
    return func,df

#===========out of the function ========================0
#tolerance
TOL=1.0e-5
#maximum number of iterations
JMAX = 20
#initial guess
x_1=0.25
x_2=0.35
x_0 = 0.5*(x_1 + x_2)
print("initial guess = {}".format(x_0))
print("function at x_0 = {}".format(func_eval(x_0)[0]))
print("derivative at x_0 = {}".format(func_eval(x_0)[1]))
# loop updating x_0
for i in range(JMAX):
    delta_x = func_eval(x_0)[0]/func_eval(x_0)[1]
    x_0 -= delta_x
    if (x_1-x_0)*(x_0-x_2) <= 0:
        print("out of brackets")
    if delta_x <= TOL:
        print("convergence. with x_0 = {}".format(x_0))
        break

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

