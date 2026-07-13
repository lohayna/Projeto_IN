#O gather sísmico CMP (Ponto Médio Comum) é um agrupamento de traços sísmicos que compartilham o mesmo ponto médio na 
# superfície entre as fontes (tiros) e os receptores (geofones ou hidrofones).

#Esse código faz um filtro F-K (frequência–número de onda) em um gather sísmico CMP. A ideia principal é transformar o dado do domínio 
# tempo–espaço (t-x) para o domínio frequência–número de onda (f-k), remover uma região específica do espectro (por exemplo, ruídos coerentes 
# como ondas lineares), aplicar uma suavização na borda do filtro e voltar para o domínio original. 

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

nt = 1001 #cada traço possui nt amostras temporais
dt = 1e-3 #o sismograma foi amostradado a cada 1 milissegundo
#o tempo total registrado sera de nt*dt 

#parametros espaciais
nr = 61 #numero de receptores
dr = 5.0 #espaçamento entre os receptores
#o comprimento total do spread é nr*dr

#parametros do filtro
#No domínio F-K, eventos lineares aparecem como retas: f=v*k v(velocidade aparente)
angle = 20 #angulo que sera removido no espectro fk
max_frequency = 60 #essa sera a referencia de freq max para desenhar a regiao do filtro, ele vai normalizar a reta que zera a energia da regiao que sera cortada

#servem para suavizar a borda do filtro para que ele faça uma transição suave e nao apareça oscilaçoes artificais
smoothing_stdv = 5.0
smoothing_samples = 2

file = f"cmp_gather_{nt}x{nr}.bin"

data = np.fromfile(file, count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

data_fk = np.fft.fftshift(np.fft.fftn(data))#fftn pra qualquer dimensao

frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
wavenumber = np.fft.fftshift(np.fft.fftfreq(nr, dr))

df = np.abs(np.abs(frequency[1]) - np.abs(frequency[0]))
dk = np.abs(np.abs(wavenumber[1]) - np.abs(wavenumber[0]))

slope = wavenumber * (max_frequency / np.max(wavenumber)) * np.tan(np.radians(angle))

# (max_frequency / np.max(wavenumber)) normalização do wavenumber para ter angulos realistas

data_fk_filtered = data_fk.copy()

for wn in range(len(wavenumber)):

    pos_slope = +slope[wn]
    neg_slope = -slope[wn]

    nb = np.logical_and(frequency < neg_slope, frequency > pos_slope)
    pb = np.logical_and(frequency < pos_slope, frequency > neg_slope)

    target = nb if pos_slope < neg_slope else pb

    data_fk_filtered[target, wn] = 0.0

    smooth = np.where(target == True)[0]

    if len(smooth) > 2*smoothing_samples:

        beg = smooth[0]
        end = smooth[-1]

        smooth = np.concatenate((np.arange(beg-1,beg-smoothing_samples-1,-1, dtype = int)[::-1], smooth))
        smooth = np.append(smooth, (np.arange(end,end+smoothing_samples, dtype = int)))

        data_fk_filtered[smooth, wn] = gaussian_filter(data_fk_filtered[smooth, wn], 5)


data_filtered = np.real(np.fft.ifftn(np.fft.ifftshift(data_fk_filtered)))

xloc = np.linspace(0, nr-1, 5, dtype = int)
xlab = np.linspace(0, nr-1, 5)*dr

tloc = np.linspace(0, nt-1, 11, dtype = int)
tlab = np.around(tloc*dt, decimals = 1)

scale = 0.9*np.std(data)

fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (10, 8))

ax[0,0].imshow(data, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
ax[0,0].set_yticks(tloc)
ax[0,0].set_yticklabels(tlab)
ax[0,0].set_xticks(xloc)
ax[0,0].set_xticklabels(xlab)
ax[0,0].set_title("Gather")
ax[0,0].set_xlabel("Offset [m]")
ax[0,0].set_ylabel("Two way time [s]")

ax[0,1].imshow(np.abs(data_fk), aspect = "auto", cmap = "jet", extent = [np.min(wavenumber),(-1)*np.min(wavenumber),np.min(frequency),(-1)*np.min(frequency)])
ax[0,1].set_ylim([-60, 60])
ax[0,1].set_title("FK domain")
ax[0,1].set_xlabel(r"Wavenumber [m$^{-1}$]")
ax[0,1].set_ylabel("Frequency [Hz]")
ax[0,1].plot(np.zeros(nt), frequency, "--k")
ax[0,1].plot(wavenumber, np.zeros(nr), "--k")

ax[0,1].plot(wavenumber, +slope, "--g")
ax[0,1].plot(wavenumber, -slope, "--r")


ax[1,0].imshow(np.abs(data_fk_filtered), aspect = "auto", cmap = "jet", extent = [np.min(wavenumber),(-1)*np.min(wavenumber),np.min(frequency),(-1)*np.min(frequency)])
ax[1,0].set_ylim([-60, 60])
ax[1,0].set_title("FK domain")
ax[1,0].set_xlabel(r"Wavenumber [m$^{-1}$]")
ax[1,0].set_ylabel("Frequency [Hz]")
ax[1,0].plot(np.zeros(nt), frequency, "--k")
ax[1,0].plot(wavenumber, np.zeros(nr), "--k")

ax[1,0].plot(wavenumber, +slope, "--g")
ax[1,0].plot(wavenumber, -slope, "--r")


ax[1,1].imshow(data_filtered, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
ax[1,1].set_yticks(tloc)
ax[1,1].set_yticklabels(tlab)
ax[1,1].set_xticks(xloc)
ax[1,1].set_xticklabels(xlab)
ax[1,1].set_title("Gather")
ax[1,1].set_xlabel("Offset [m]")
ax[1,1].set_ylabel("Two way time [s]")

fig.tight_layout()
plt.show()