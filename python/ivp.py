# python3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# x''(t) + c*x'(t) + k*x(t) = 0
# x'(t) = v(t)
# v'(t) = -c*v(t) - k*x(t)

def spring(t,y,c,k):
    x, v = y
    dxdt = [v, -c*v - k*x]
    return dxdt

# constants values
c = 0.25
k = 5.0

# initial values
y0 = [0.1, 0.0]

# time space
t = np.linspace(0,10,101)

sol = solve_ivp(spring, [0,10], y0, t_eval=t, args=(c,k))

print(sol.t)
print(sol.y)

plt.plot(sol.t, sol.y[0,:], 'b', label='x(t)')
plt.plot(sol.t, sol.y[1,:], 'g', label='v(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
plt.close('all')

