#wave propagation 1D
##bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def wavelet(t,fm):

 return (1-2*(np.pi*t*fm)**2)*np.exp(-(np.pi*t*fm)**2)

def get_damp(nz, nb, factor):
  nzz = nz + 2*nb

  damp_z = np.zeros(nzz)

  for i in range(nzz):

    if nb <= i < nb + nz:
      damp_z[i] = 1.0

    elif i < nb:
      d = nb - i
      damp_z[i] = np.exp(-(factor * d) * (factor * d))

    else:
      d = i - (nb + nz - 1)
      damp_z[i] = np.exp(-(factor * d) * (factor * d))

  return damp_z

# Modelo
nz = 201
dz = 5
c = 1500.0

model = np.full(nz, c)
model[50:] = 2500.0

plt.plot(model)
plt.show()

# Geometria
offset = 5
sz = 0

# Parametros de Modelagem
nt = 2001
dt = 1e-3

# Wavelet
fm = 15 # Hz
tlag = 0.15
t = np.arange(nt)*dt - tlag

ricker = wavelet(t, fm)

# Modelagem
nb = 100
factor = 0.0015
nzz = nz + 2*nb

# Extent Model
model_ext = np.zeros(nzz)
model_ext[:nb] = model[0]
model_ext[nb:nz+nb] = model
model_ext[nb+nz:] = model[-1]

damp_z = get_damp(nz, nb, factor)

nsnaps = 51
snap_ratio = int(nt / nsnaps) + 1
snapshots = np.zeros((nsnaps, nzz))
snap_id = 0

# Matriz de Pressao
P = np.zeros((nzz, nt))

for t in range(1, nt - 1):
  src_z = sz + nb
  P[src_z, t] += ricker[t]
 
  # calculate spatial and temporal derivatives
  for i in range(1, nzz - 1):
    laplacian = (P[i+1, t] - 2 * P[i, t] + P[i-1, t]) / dz**2

    P[i, t+1] = (dt * model_ext[i])**2 * laplacian + 2*P[i, t] - P[i, t-1]

  # applying border condition
  for i in range(1, nzz - 1):
    P[i, t+1] *= damp_z[i]
    P[i, t] *= damp_z[i]

  # capturing snapshots
  if not t % snap_ratio:
    snapshots[snap_id] = P[:, t].copy()
    snap_id += 1

# plot result

plt.imshow(P, aspect="auto", cmap="Greys")
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


