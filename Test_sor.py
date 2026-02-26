import numpy as np
import matplotlib.pyplot as plt
from SRC.SOR import SORsolver

def test_linear_solution():
    solver = SORsolver(N=100, omega=1.8, tol=1e-6, max_iter=10000)
    #solver.object_mask[20:30, 20:30] = True
    center = solver.N // 2
    solver.object_mask[center, center] = True
    c = solver.solve(verbose=True)

    # Plot 2D heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(c.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
    #plt.imshow(c.T, origin="lower", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(label="Concentration")
    plt.title("Laplace with Central Square Sink")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    test_linear_solution()
