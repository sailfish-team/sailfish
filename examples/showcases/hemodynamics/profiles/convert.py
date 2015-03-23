import numpy as np
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt

orig = np.loadtxt('basilar_valencia_alvaro.dat')
gf = interpolate.interp1d(orig[:,0], orig[:,1], kind='linear')
x = np.linspace(orig[0,0], orig[-1,0], 200)
plt.plot(x, gf(x))
period = orig[-1,0] - orig[0,0]
print 'period = %e' % period
data = np.vstack((x[:-1] - x[0], gf(x[:-1]))).transpose()
np.savetxt('basilar_valencia_profile_uniform_lin_int.dat', data[:100,:])
