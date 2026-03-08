import numpy as np
import matplotlib.pyplot as plt

# Choses parameters for the Gray-Scott model
dt = 1.0
dx = 1.0
Du = 0.16
Dv = 0.08
f = 0.035
k = 0.060
random_starting_noise = False

N = 100
steps = 5000
noise = 0.01


def neumann(Z):
    Z[0, :] = Z[1, :]
    Z[-1, :] = Z[-2, :]
    Z[:, 0] = Z[:, 1]
    Z[:, -1] = Z[:, -2]
    return Z


def laplacian_neumann(Z, dx):
    L = np.zeros_like(Z)

    L[1:-1, 1:-1] = (
        Z[2:, 1:-1] +
        Z[:-2, 1:-1] +
        Z[1:-1, 2:] +
        Z[1:-1, :-2] -
        4 * Z[1:-1, 1:-1]
    ) / dx**2

    return L


# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

# Set U to 0.5 everywhere.
u = 0.5 * np.ones((N, N))


# Set V to 0 everywhere, except for a small square of size r
# in the middle where V is set to 0.25.
v = np.zeros((N, N))
r = 10
middle = N // 2
v[middle-r:middle+r, middle-r:middle+r] = 0.25


if random_starting_noise:
    # Add random noise to starting condition.
    u += noise * (np.random.rand(N, N) - 0.5)
    v += noise * (np.random.rand(N, N) - 0.5)

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)


    u = neumann(u)
    v = neumann(v)

save_times = [1, 500, 1000, 2000, 5000]
saved_u = {}
saved_v = {}


for n in range(1, steps + 1):

    u = neumann(u)
    v = neumann(v)

    Lu = laplacian_neumann(u, dx)
    Lv = laplacian_neumann(v, dx)

    reaction = u * v**2

    u_new = u + dt * (Du * Lu - reaction + f * (1 - u))
    v_new = v + dt * (Dv * Lv + reaction - (f + k) * v)

    u = np.clip(u_new, 0, 1)
    v = np.clip(v_new, 0, 1)

    if n in save_times:
        saved_u[n] = u.copy()
        saved_v[n] = v.copy()


fig, axes = plt.subplots(2, len(save_times), figsize=(18, 8))

fig.suptitle(
    f"Gray-Scott Model\n"
    f"Du={Du}, Dv={Dv}, f={f}, k={k}, dt={dt}",
    fontsize=16
)

for j, t in enumerate(save_times):

    im1 = axes[0, j].imshow(saved_u[t], cmap="viridis")
    axes[0, j].set_title(f"U concentration\n t = {t}")
    axes[0, j].axis("off")
    plt.colorbar(im1, ax=axes[0, j], fraction=0.046)

    im2 = axes[1, j].imshow(saved_v[t], cmap="viridis")
    axes[1, j].set_title(f"V concentration\n t = {t}")
    axes[1, j].axis("off")
    plt.colorbar(im2, ax=axes[1, j], fraction=0.046)

plt.tight_layout()
plt.show()