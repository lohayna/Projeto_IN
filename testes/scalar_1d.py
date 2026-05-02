#wave propagation 1D
##bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def wavelet(t,fm):

 return (1-2*(np.pi*t*fm)**2)*np.exp(-(np.pi*t*fm)**2)

# Parametros do Modelo
nz = 201
dz = 5

# Geometria
offset = 5
receivers = np.arange(0, nz, offset)
sz = nz // 2

nrec = len(receivers)

# Parametros de Modelagem
nt = 1001
dt = 1e-3

# Wavelet
fm = 15 # Hz
tlag = 0.15
t = np.arange(nt)*dt - tlag

ricker = wavelet(t, fm)

# Matriz de Pressao
P = np.zeros((nz, nt))

# Velocidade do meio
c = 1500.0

# Modelagem
nsnaps = 51
snap_ratio = int(nt / nsnaps) + 1
snapshots = np.zeros((nsnaps, nz))
snap_id = 0

seismogram = np.zeros((nt, nrec))

for t in range(1, nt - 1):
  P[sz, t] += ricker[t]
  
  for i in range(1, nz - 1):
    laplacian = (P[i+1, t] - 2 * P[i, t] + P[i-1, t]) / dz**2

    P[i, t+1] = (dt * c)**2 * laplacian + 2*P[i, t] - P[i, t-1]

  for irec in range(nrec):
    seismogram[t, irec] = P[receivers[irec], t]

  if not t % snap_ratio:
    snapshots[snap_id] = P[:, t].copy()
    snap_id += 1

# plot result

plt.imshow(seismogram, aspect="auto", cmap="Greys")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

ims = []
for snap in snapshots:
  snap_frame = ax.plot(snap, color='b')

  ims.append(snap_frame)

ani = animation.ArtistAnimation(
  fig, ims,
  interval=100,
  blit=False,
  repeat_delay=0
)

#writer = animation.FFMpegWriter(fps=20, bitrate=1800)
#ani.save("animacao.mp4", writer=writer)

plt.show()

# TODO:
# Aplicar Condicao de Contorno
