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

#wav  = ricker(f, t)
#wav = sinc(t)
F_wav = np.fft.rfft(wav)
freq = np.fft.rfftfreq(nt, dt)
wav_ormsby = ormsby(t, f, f1, f2, f3)
wav_klauder = klauder(t, f, f1, T)

plt.figure(figsize=(8,10))

plt.subplot(3,1,1)
plt.plot(t, wav)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Ricker")
plt.grid()

plt.subplot(3,2,1)
wav  = ricker(f, t)
F_wav = np.fft.rfft(wav)
freq = np.fft.rfftfreq(nt, dt)
plt.figure()
plt.plot(freq, np.abs(F_wav*F_wav))
plt.title("Ricker")
plt.grid()
plt.show()

plt.subplot(3,1,2)
plt.plot(t, wav_ormsby)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Ormsby")
plt.grid()

plt.subplot(3,1,3)
plt.plot(t, wav_klauder)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Klauder")
plt.grid()

plt.tight_layout()
plt.show()