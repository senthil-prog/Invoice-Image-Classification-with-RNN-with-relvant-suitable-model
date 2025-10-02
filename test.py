import matplotlib.pyplot as plt
import numpy as np

print("Testing graph display...")

# Simple test graph
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Graph - Close this window to continue")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print("✓ Graph test complete!")