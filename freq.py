import numpy as np
import matplotlib.pyplot as plt

def ricker(f, t):
    return (1-2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)

def ricker_frcorte(fc, t):

    f = fc/3*np.sqrt(np.pi)
    to = 0 #2*np.sqrt(np.pi)/fc
    td = t - to

    return (1-2*np.pi**2*f**2*td**2)*np.exp(-np.pi**2*f**2*td**2)

nt = 1000
t = np.linspace(-1,1,nt)
dt = t[1] - t[0]
fc = 50
f = 50

wav  = ricker(f, t)
wav_cor  = ricker_frcorte(fc, t)

F_wav = np.fft.fft(wav)
F_wav_cor = np.fft.fft(wav_cor)

freq = np.fft.fftfreq(nt, dt)

# Apenas frequências positivas
mask = freq >= 0

# Figura
plt.figure(figsize=(12,8))

# Ricker
plt.subplot(2,2,1)
plt.plot(t, wav)
plt.title("Ricker")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()

# Ricker por frequência de corte
plt.subplot(2,2,2)
plt.plot(t, wav_cor)
plt.title("Ricker (frequência de corte)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid()

# Espectro da Ricker
plt.subplot(2,2,3)
plt.plot(freq[mask], np.abs(F_wav)[mask])
plt.title("Espectro - Ricker")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|FFT|")
plt.grid()

# Espectro da frequência de corte
plt.subplot(2,2,4)
plt.plot(freq[mask], np.abs(F_wav_cor)[mask])
plt.title("Espectro - Ricker (frequência de corte)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("|FFT|")
plt.grid()

plt.tight_layout()
plt.show()

