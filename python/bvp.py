# python3
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# EIw''(x) + M(x) = 0
# w'(x) = theta(x)
# theta'(x) = -M(x)/EI

E = 50.0
I = (30.0*3.0**3)/12.0
EI = E*I
length = 30.0
force = 0.015*30.0**2

def moment(x):
    M = np.zeros((x.shape[0]), dtype=np.float64)
    for i in range(x.shape[0]):
        if x[i]<=0.5*length:
            M[i]=-0.5*force*x[i]
        else:
            M[i]=-0.5*force*(length-x[i])
    return M

def beam(x,y):
    dydx = [y[1], -moment(x)/(EI)]
    return dydx

def beam_bc(ya, yb):
    return [ya[0], yb[0]]

# arch length
x = np.linspace(0,length,11)

# boundary values
y = np.zeros((2, x.size))

sol = solve_bvp(beam, beam_bc, x, y)

print(sol.x)
print(sol.y)

plt.plot(sol.x, sol.y[0,:], 'b', label='y(x)')
plt.plot(sol.x, sol.y[1,:], 'g', label='theta(x)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
plt.close('all')

