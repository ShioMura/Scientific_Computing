import numpy as np
import matplotlib.pyplot as plt
from SRC.SOR import SORsolver

def test_linear_solution():
    solver = SORsolver(N=10, omega=1.8, tol=1e-6, max_iter=10000)
    c = solver.solve(verbose=True)
    # Check if the solution is close to zero everywhere (since boundary conditions are zero)

    y = np.linspace(0, 1, solver.N)
    analytical = y 

    numerical_profile = np.mean(c, axis=0)  # Average across rows to get a profile along y

    plt.plot(y, analytical, label="Analytical (c=y)")
    plt.plot(y, numerical_profile, "--", label="Numerical (SOR)")
    plt.xlabel("y")
    plt.ylabel("c")
    plt.legend()
    plt.title("Laplace Equation Solution")
    plt.show()


if __name__ == "__main__":
    test_linear_solution()
