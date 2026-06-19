import numpy as np
import matplotlib.pyplot as plt

def ricker(f, t):
    return (1-2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)

def ricker_frcorte(fc, t):

    f = fc/3*np.sqrt(np.pi)
    to = 2*np.sqrt(np.pi)/fc
    td = t - to

    return (1-2*np.pi**2*f**2*td**2)*np.exp(-np.pi**2*f**2*td**2)

nt = 1000
t = np.linspace(-1,1,nt)
fc = 30
f = 50

wav  = ricker(f, t)
wav_cor  = ricker_frcorte(fc, t)

plt.subplot(2,2,1)

plt.plot(wav, label=f"{f}Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Ricker")
plt.legend()
plt.grid()

plt.subplot(2,2,2)

plt.plot(wav_cor, label=f"{fc}Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Ricker Corte")
plt.grid()

plt.show()

