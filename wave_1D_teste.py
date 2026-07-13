#wave propagation 1D
##bibliotecas

import numpy as np
import matplotlib.pyplot as plt

#criar modelo
n = 1000
velocity = np.zeros(n)
velocity[0:400] = 2.3
velocity[400:800] = 2.6
velocity[800:1000] = 2.9

#plt.plot(np.arange(n), velocity)
plt.plot(velocity, np.arange(nz))

plt.gca().invert_yaxis()
plt.show()

#criar geometria
dx = 5
dz = 5
nx = 200
nz = 200

sx = 0
sz = 1

#ricker

def f(t,fm):

 return (1-2*(np.pi*t*fm)**2)*np.exp(-(np.pi*t*fm)**2)

fm = 5
t = np.linspace(-1, 1, 1000)

ricker = f(t, fm)

#DIFERENÇAS FINITAS

#def wave(x, t, v, A, L):
    