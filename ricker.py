import numpy as np
import matplotlib.pyplot as plt
 
def ricker(f, t):
    return (1-2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)

t = np.linspace(-1,1,1000)
f = 25

wav  = ricker(f, t)

plt.figure(figsize=(8,5))
plt.plot(t, wav)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Ricker")
plt.grid()
plt.show()