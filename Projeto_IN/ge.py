import numpy as np
import matplotlib.pyplot as plt
 
def ricker(f, t):
    return (1-2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)

def sinc(x):
    return np.sinc(x / np.pi)  # sinc(x) = sin(x)/x

def ormsby(t, f, f1, f2, f3):
    
    term1 = ((np.pi * f3)**2 / (np.pi * f3 - np.pi * f2)) * (sinc(np.pi * f3 * t))**2
    term2 = ((np.pi * f2)**2 / (np.pi * f3 - np.pi * f2)) * (sinc(np.pi * f2 * t))**2
    
    term3 = ((np.pi * f1)**2 / (np.pi * f1 - np.pi * f)) * (sinc(np.pi * f1 * t))**2
    term4 = ((np.pi * f)**2 / (np.pi * f1 - np.pi * f)) * (sinc(np.pi * f * t))**2

    return (term1 - term2) - (term3 - term4)

def klauder(t, f, f1, T):
    
    k = (f1 - f / T)   # taxa de variação da frequência
    f0 = (f1 + f / 2) # frequência central
    x = np.pi * k * t # evitar divisão por zero

    klauder = np.real(
        (np.sin(np.pi * k * t * (T - t)) / (x + 1e-10))
        * np.exp(2j * np.pi * f0 * t)
    )

    return klauder

nt = 1000
t = np.linspace(-1,1,nt)
dt = t[1]-t[0]
T = 0.5
f = 50
f1, f2, f3 = 20, 10, 5

wav  = ricker(f, t)
F_wav = np.fft.fft(wav)
freq = np.fft.fftfreq(nt, dt)

wav_ormsby = ormsby(t, f, f1, f2, f3)
F_wav_ormsby = np.fft.fft(wav_ormsby)

wav_klauder = klauder(t, f, f1, T)
F_wav_klauder = np.fft.fft(wav_klauder)

plt.figure(figsize=(10,8))

#ricker

plt.subplot(3,2,1)

plt.plot(wav)
plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Ricker")
plt.grid()

plt.subplot(3,2,2)

plt.plot(freq, np.abs(F_wav))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Ricker Spectrum")
plt.grid()

#ormsby

plt.subplot(3,2,3)

plt.plot(wav_ormsby)
plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Ormsby")
plt.grid()

plt.subplot(3,2,4)

plt.plot(freq, np.abs(F_wav_ormsby))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Ormsby Spectrum")
plt.grid()

#klauder

plt.subplot(3,2,5)

plt.plot(wav_klauder)
plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.title("Klauder")
plt.grid()

plt.subplot(3,2,6)

plt.plot(freq, np.abs(F_wav_klauder))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Klauder Spectrum")
plt.grid()

plt.tight_layout()
plt.show()