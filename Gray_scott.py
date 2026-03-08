import numpy as np
import matplotlib.pyplot as plt

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


def Gray_Scott(dt=1.0, dx=1.0, Du=0.16, Dv=0.08, f=0.035, k=0.060,
               random_starting_noise=True,
               save_times=None):

    N = 100
    steps = 5000
    noise_amp = 0.01

    if save_times is None:
        save_times = [1, 500, 1000, 2000, 5000]

    # Initial conditions
    u = 0.5 * np.ones((N, N))
    v = np.zeros((N, N))

    # Square of size r.
    r = 10
    c = N // 2
    v[c-r:c+r, c-r:c+r] = 0.25

    if random_starting_noise:
        u += noise_amp * (np.random.rand(N, N) - 0.5)
        v += noise_amp * (np.random.rand(N, N) - 0.5)

    saved_u = {}
    saved_v = {}

    # Save initial state if requested
    if 0 in save_times:
        saved_u[0] = u.copy()
        saved_v[0] = v.copy()

    # Time loop
    for n in range(1, steps + 1):
        u = neumann(u)
        v = neumann(v)

        Lu = laplacian_neumann(u, dx)
        Lv = laplacian_neumann(v, dx)

        reaction = u * v**2

        u = u + dt * (Du * Lu - reaction + f * (1 - u))
        v = v + dt * (Dv * Lv + reaction - (f + k) * v)

        # Optional clipping for numerical stability
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        if n in save_times:
            saved_u[n] = u.copy()
            saved_v[n] = v.copy()

    return u, v, saved_u, saved_v


# --------------------------------------------------
# Parameters
# --------------------------------------------------
dt = 1.0
dx = 1.0
Du = 0.16
Dv = 0.08
f = 0.035
k = 0.060

# --------------------------------------------------
# Plot time evolution for one parameter set
# --------------------------------------------------
save_times = [1, 500, 1000, 2000, 5000]

u, v, saved_u, saved_v = Gray_Scott(
    dt=dt, dx=dx, Du=Du, Dv=Dv, f=f, k=k,
    random_starting_noise=True,
    save_times=save_times
)

fig, axes = plt.subplots(2, len(save_times), figsize=(18, 8))

fig.suptitle(
    f"Gray-Scott Model\nDu={Du}, Dv={Dv}, f={f}, k={k}, dt={dt}",
    fontsize=16
)

for j, t in enumerate(save_times):
    im1 = axes[0, j].imshow(saved_u[t], cmap="viridis")
    axes[0, j].set_title(f"U concentration\n t = {t}")
    axes[0, j].axis("off")
    plt.colorbar(im1, ax=axes[0, j], fraction=0.046)

    im2 = axes[1, j].imshow(saved_v[t], cmap="magma")
    axes[1, j].set_title(f"V concentration\n t = {t}")
    axes[1, j].axis("off")
    plt.colorbar(im2, ax=axes[1, j], fraction=0.046)

plt.tight_layout()
plt.savefig("gray_scott_evolution_initial.png", dpi=300)

# --------------------------------------------------
# Vary k while keeping f fixed
# --------------------------------------------------
f_fixed = 0.035
k_values = [0.050, 0.055, 0.060, 0.065]

fig, axes = plt.subplots(1, len(k_values), figsize=(18,5))

for i, k_val in enumerate(k_values):
    u, v, _, _ = Gray_Scott(
        dt=dt, dx=dx, Du=Du, Dv=Dv,
        f=f_fixed, k=k_val,
        random_starting_noise=True
    )

    im = axes[i].imshow(v, cmap="magma")
    axes[i].set_title(f"f={f_fixed}, k={k_val}")
    axes[i].axis("off")

plt.suptitle("Effect of varying the removal rate in Gray_scott model, while keeping the rest of the parameters constant.")
plt.tight_layout()
plt.savefig("gray_scott_varying_k.png", dpi=300)

# --------------------------------------------------
# Vary f while keeping k fixed
# --------------------------------------------------
k_fixed = 0.060
f_values = [0.030, 0.035, 0.040, 0.045]

fig, axes = plt.subplots(1, len(k_values), figsize=(18,5))

for i, f_val in enumerate(f_values):
    u, v, _, _ = Gray_Scott(
        dt=dt, dx=dx, Du=Du, Dv=Dv,
        f=f_val, k=k_fixed,
        random_starting_noise=True
    )

    im = axes[i].imshow(v, cmap="magma")
    axes[i].set_title(f"f={f_val}, k={k_fixed}")
    axes[i].axis("off")

plt.suptitle("Effect of varying feedrate in the Gray-Scott model, while keeping rest of the parameters constant.")
plt.tight_layout()
plt.savefig("gray_scott_varying_f.png", dpi=300)