import numpy as np
import matplotlib.pyplot as plt

a_values = [32, 31.9, 31.8, 32.1, 32.2]

eigenvalues = []

for a in a_values:
    A = np.array([[-6, 28, 21], [4, -15, -12], [-8, a, 25]])
    
    eigenvals = np.linalg.eigvals(A)
    eigenvalues.append(eigenvals)

    t_values = np.linspace(0, 3, 400)

    def characteristic_polynomial(t):
        return np.linalg.det(A - t * np.identity(3))

    poly_values = [characteristic_polynomial(t) for t in t_values]
    plt.plot(t_values, poly_values, label=f'a={a}')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('t')
plt.ylabel('Characteristic Polynomial p(t)')
plt.legend()
plt.title('Characteristic Polynomial vs. t for Various a Values')
plt.grid()

for i, a in enumerate(a_values):
    print(f'Eigenvalues for a={a}: {eigenvalues[i]}')

# Show the plots
plt.show()
