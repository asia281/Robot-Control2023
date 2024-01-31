import numpy as np

def check_stability(alpha):
    A = np.array([[10 - alpha, -3], [-3, -100]])
    eigenvalues, _ = np.linalg.eigh(A)
    return np.all(eigenvalues < 0)

# Now you can find the smallest positive alpha for which the system is stable.
alpha = 0.01  # Initialize with a small positive value
while not check_stability(alpha):
    alpha += 0.01

while check_stability(alpha):
    alpha -= 0.0000001

print("Smallest stable alpha:", alpha)