import numpy as np
import matplotlib.pyplot as plt


class MonteCarloDLA:
    def __init__(self, N=100, p_stick=1.0):
        self.N = N
        self.p_stick = p_stick
        self.cluster = np.zeros((N, N), dtype=bool)

        # Seed at center
        center = N // 2
        self.cluster[center, center] = True

    def release_particle(self):
        x = np.random.randint(0, self.N)
        y = self.N - 1
        return x, y

    def is_adjacent_to_cluster(self, x, y):
        if x <= 0 or x >= self.N - 1 or y <= 0 or y >= self.N - 1:
            return False

        return (
            self.cluster[x+1, y] or
            self.cluster[x-1, y] or
            self.cluster[x, y+1] or
            self.cluster[x, y-1]
        )
    def random_walk(self):
        x, y = self.release_particle()

        max_steps = 10000
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                return False
            # Random move
            direction = np.random.randint(4)

            if direction == 0:
                x += 1
            elif direction == 1:
                x -= 1
            elif direction == 2:
                y += 1
            else:
                y -= 1

            # Keep inside domain
            x = max(1, min(self.N - 2, x))
            y = max(1, min(self.N - 2, y))

            # Check sticking
            if self.is_adjacent_to_cluster(x, y):
                if np.random.rand() < self.p_stick:
                    self.cluster[x, y] = True
                    return True

    def run(self, particles=500):
        for i in range(particles):
            self.random_walk()

    def plot(self):
        plt.figure(figsize=(6,6))
        plt.imshow(self.cluster.T, origin="lower", cmap="binary")
        plt.title(f"Monte Carlo DLA (p = {self.p_stick})")
        plt.show()


if __name__ == "__main__":

    particles = 300
    p_values = [1.0, 0.5, 0.2]

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    for ax, p in zip(axes, p_values):

        print(f"Running Monte Carlo DLA with p = {p}")

        dla = MonteCarloDLA(N=100, p_stick=p)
        dla.run(particles=particles)

        ax.imshow(dla.cluster.T, origin="lower", cmap="binary")
        ax.set_title(f"p = {p}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Monte Carlo DLA Comparison")
    plt.tight_layout()
    plt.show()