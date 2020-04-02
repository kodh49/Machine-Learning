#import module
import numpy as np
import scipy.misc as cp
from scipy.constants import *
import matplotlib.pyplot as plt

# define global constants
e = np.e # euler's constant
pi = np.pi # pi
k = Boltzmann # Boltzmann constant
T = 1.37 # Temperature at the center of the sun in keV

# define tunneling probability function
def quantum_prob(E): # E is presented as keV
    b = 493.5
    return pow(e, -np.sqrt(b/E))

# define Maxwell-boltzmann energy distribution function
def maxwell_boltzmann(E): # E is presented as keV
    b = E/(T) # exponential factor
    return pow(1/e, b)

# define a function to optimize
def gamow(E): # E is presented as keV
    return (quantum_prob(E)*maxwell_boltzmann(E))


x_init = 0.1 # initial x value
lr = 1e-5 # Learning rate
while (True):
    z = gamow(x_init)/cp.derivative(gamow, x_init, dx=1e-5, n=1)
    if (z < 0): # Where df/dx reaches 0
        break
    else:
        x_init = x_init + lr # Update x var
        print("x = "+str(x_init))
        continue

# Graph plot
xval = np.linspace(0, 10, 10000)
yval = gamow(xval)
zval = gamow(xval)/cp.derivative(gamow, xval, dx=1e-5, n=1)
plt.plot(xval, zval)
plt.show()

# Expected Value of Gamow Peak / results
val = ((np.sqrt(493.5)*T)/2)**(2/3)
err = 100*(np.abs(val - x_init)/val)
print("Approximated value (keV) = "+str(x_init))
print("Expected value (keV) = "+str(val))
print("Error (%) = "+str(err))