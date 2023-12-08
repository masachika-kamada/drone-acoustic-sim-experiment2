import matplotlib.pyplot as plt
import numpy as np


estimated_angles = np.load("estimated_angles.npy")
print(estimated_angles.shape)
music_spectrum = np.load("music_spectrum.npy")

color = (0.0, 0.0, 1.0)  # Color of the lines
alpha = 0.2  # Transparency level

for j in range(36):
    i = j * 10
    vals = np.concatenate([music_spectrum[i:], music_spectrum[:i]])
    plt.polar(estimated_angles, vals, color=color, alpha=alpha)

plt.grid(True)

plt.show()
plt.close()
