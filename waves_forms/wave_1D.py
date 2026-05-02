#wave propagation 1D
##bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def wavelet(t,fm):

 return (1-2*(np.pi*t*fm)**2)*np.exp(-(np.pi*t*fm)**2)

#criar modelo
nz = 1000
velocity = np.zeros(nz)
velocity[0:400] = 2300
velocity[400:800] = 2600
velocity[800:1000] = 2900

#plt.plot(np.arange(n), velocity)
#plt.plot(velocity, np.arange(nz))

#plt.gca().invert_yaxis()
#plt.show()

#criar geometria
dx = 5
dz = 5
recz = 100 #meus receptor

sz = 25
prof = np.arange(nz)*dz
#ricker

nt = 1001
dt = 1e-3

fm = 15
tlag = 0.15
t = np.arange(nt)*dt - tlag

ricker = wavelet(t, fm)

plt.plot(np.arange(nt), ricker)
plt.show()

# Pressure

P = np.zeros((nz, nt))

c = 1500

laplacian = np.zeros(nz)
for t in range(1, nt - 1):
    P[sz, t] += ricker[t]
    
    for i in range(1, nz - 1):
        laplacian[i] = (P[i+1, t] - 2 * P[i, t] + P[i-1, t]) / dz**2
        
    P[:, t+1] = (dt * c)**2 * laplacian + 2*P[:, t] - P[:, t-1]
    
    

def wave_animation(P, nt):
    fig, ax = plt.subplots(num="Wavefield plot", figsize=(8, 8), clear=True)
    
    im, = ax.plot(prof,P[:,0])

    ax.set_title("Wavefield plot", fontsize=18)
    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)

    def animate(frame):
        im.set_ydata(P[:, frame])
        ax.set_title(f"time = {frame*dt:.3f} s")
        return im,

    ani = FuncAnimation(fig, animate, frames=range(nt), interval=10)
    ani.save("wave_propagation.gif", writer=PillowWriter(fps=20))
    
    plt.tight_layout()
    plt.show()

wave_animation(P, nt)

    