#python3
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

dof = 2

synth_data = t.rvs(df=dof, loc=28.2, scale=3.5, size=10)

data_mean = np.mean(synth_data)
data_std = np.std(synth_data, ddof=1)

print(synth_data)
print(data_mean)
print(data_std)

dist = t(df=dof, loc=data_mean, scale=data_std)
interval = dist.interval(0.90)
p_value = dist.cdf(35.0)

print(interval)
if p_value>0.5:
    prob = 2.0*(1.0-p_value)
else:
    prob = 2.0*p_value
print(prob)

x = np.linspace(dist.ppf(0.05),dist.ppf(0.95), 100)

fig, ax = plt.subplots(1, 1)
ax.plot(x, dist.pdf(x), 'r-', lw=3, alpha=0.6, label='t pdf')
ax.hist(synth_data, density=True, bins='auto', histtype='stepfilled')
plt.show()

from scipy.stats import f_oneway
synth_data1 = t.rvs(df=dof, loc=35.1, scale=2.6, size=10)
data_mean1 = np.mean(synth_data1)
data_std1 = np.std(synth_data1, ddof=1)

print(synth_data1)
print(data_mean1)
print(data_std1)

anova = f_oneway(synth_data, synth_data1)

print(anova)
