##bibliotecas

import dataclasses
from plistlib import FMT_BINARY
import numpy as np
import matplotlib.pyplot as plt

##Input pulse(f amplitude da Ricker)
#ricker equation

def f(t,fm):

 return (1-2*(np.pi*t*fm)**2)*np.exp(-(np.pi*t*fm)**2)

fm = 5
t = np.linspace(-1, 1, 1000)

ricker = f(t, fm)

##Reflectivy function
#Geologia section
#depth = [0, 10, 80, 100, 200]   
#density = {"Clay": 2.6, "]Sandstone": 2.3, "Dolomite": 2.9}
n = 1000
density = np.zeros(n)
#velocity = {"Clay": 2.5, "Sandstone": 6, "Dolomite": 6.5}
velocity = np.zeros(n)

density[0:200] = 2.6
density[200:600] = 2.3
density[600:1000] = 2.9

velocity[0:200] = 2.5
velocity[200:600] = 6
velocity[600:1000] = 6.5

#Reflectivity Fuction

#n=500
#R = np.zeros(n)

def R(density, velocity):
    R = np.zeros(len(density) - 1)
    for i in range(len(density) - 1):
        Z1 = density[i] * velocity[i]
        Z2 = density[i+1] * velocity[i+1]
        R[i] = (Z2 - Z1) / (Z2 + Z1)
    return R
    
Refletivitidade = R(density, velocity)
print(R(density, velocity))
depth = np.arange(0, n*1) 

# Convolution(refletivity * ricker)
#'full' → convolução completa
#same' → saída com mesmo tamanho do sinal 
#valid' → só onde há sobreposição completa

Conv = np.convolve(Refletivitidade, ricker, mode= "same")
#depth_conv = np.linspace(min(depth), max(depth), len(Conv))

#plot of ricker

plt.subplot(1,5,1)
plt.plot(ricker, t, color="black")
#plt.xlabel("Amplitude")
#plt.ylabel("Tempo")
plt.title("Ricker")
plt.gca().invert_yaxis()
plt.grid()

#print(density["Clay"])
#step(x, y, [fmt], *, data=None, where='pre', **kwargs)

plt.subplot(1,5,2)
plt.step(density, depth, where='pre', color="black")
#plt.xlim(0, 20)   # eixo x vai de 2 até 10
#plt.ylim(0, 500)
#plt.xlabel("density")
#plt.ylabel("depth")
plt.title("Density")
plt.gca().invert_yaxis()
plt.grid()   

plt.subplot(1,5,3)    
plt.step(velocity, depth, where='pre', color="black")
#plt.xlim(0, 10)   # eixo x vai de 2 até 10
#plt.ylim(0, 200)
#plt.xlabel("velocity")
#plt.ylabel("Depth")
plt.title("Velocity")
plt.gca().invert_yaxis()
plt.grid() 

plt.subplot(1,5,4)
plt.plot([0, 0], [min(depth), max(depth)], color='black')

for depthi, Ri in zip(depth[1:], Refletivitidade):
    plt.plot([0, Ri], [depthi, depthi], color='black')

plt.gca().invert_yaxis()
plt.title("Reflectivity")
plt.grid()

plt.subplot(1,5,5)
plt.plot(Conv, depth, color="black")
#plt.xlabel("Amplitude")
#plt.ylabel("Tempo")
plt.title("Seismic Trace")
plt.gca().invert_yaxis()
plt.grid()

plt.show()