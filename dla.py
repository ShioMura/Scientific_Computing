import numpy as np
import matplotlib.pyplot as plt
from SRC.SOR import SORsolver


class DLAsolver(SORsolver):
    """
    Harmonic-measure Diffusion Limited Aggregation (DLA)

    Growth probability:
        p(i,j) ∝ |c(i,j)|^eta
    """

    def __init__(self, N=100, omega=1.8, tol=1e-6, max_iter=10000, eta=1.0):
        super().__init__(N, omega, tol, max_iter)

        self.eta = eta

        # Seed particle at center
        center = N // 2
        self.object_mask[center, center] = True

    # -------------------------------------------------
    # Find sites adjacent to aggregate
    # -------------------------------------------------
    def get_boundary_sites(self):
        boundary = []

        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):

                if self.object_mask[i, j]:
                    continue

                if (
                    self.object_mask[i + 1, j]
                    or self.object_mask[i - 1, j]
                    or self.object_mask[i, j + 1]
                    or self.object_mask[i, j - 1]
                ):
                    boundary.append((i, j))

        return boundary

    # -------------------------------------------------
    # Grow one particle using harmonic measure
    # -------------------------------------------------
    def grow_one_particle(self):

        # Solve Laplace equation
        c = self.solve()

        boundary = self.get_boundary_sites()
        if not boundary:
            return False

        # Use absolute value to avoid negative probabilities
        values = []

        for (i, j) in boundary:
            grad = np.sqrt(
        (c[i+1, j] - c[i-1, j])**2 +
        (c[i, j+1] - c[i, j-1])**2
    )
        values.append(max(grad, 1e-12) ** self.eta)

        values = np.array(values)

        total = values.sum()
        if total == 0:
            return False

        probs = values / total

        # Choose growth site
        idx = np.random.choice(len(boundary), p=probs)
        i, j = boundary[idx]

        self.object_mask[i, j] = True

        return True

    # -------------------------------------------------
    # Run DLA simulation
    # -------------------------------------------------
    def run(self, steps=200):

        for s in range(steps):
            print(f"Growing particle {s+1}/{steps}")
            if not self.grow_one_particle():
                break

    # -------------------------------------------------
    # Plot aggregate
    # -------------------------------------------------
    def plot(self):

        plt.figure(figsize=(6, 6))
        plt.imshow(self.object_mask.T, origin="lower", cmap="binary")
        plt.title(f"Harmonic Measure DLA (η = {self.eta})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    etas = [0.5, 1.0, 2.0]
    N = 120
    steps = 300

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, eta in zip(axes, etas):

        print(f"\nRunning DLA with eta = {eta}")

        dla = DLAsolver(N=N, eta=eta)
        dla.run(steps=steps)

        ax.imshow(dla.object_mask.T, origin="lower", cmap="binary")
        ax.set_title(f"η = {eta}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Harmonic Measure DLA Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("dla_eta_comparison.png", dpi=300)
    plt.show()