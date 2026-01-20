import numpy as np
import matplotlib.pyplot as plt


def plot_gp_with_subsamples(
    Y_full,
    mesh_size,
    subsample_idx,
    Y_subsampled,
    title="GP Surface with Subsampled Data",
    figsize=(5, 5),
    cmap="viridis",
    cbar_shrink=0.6,
    cbar_aspect=20,
    cbar_pad=0.1,
    dpi=150,
    vmin=None,
    vmax=None,
):
    """Plot 3D GP surface with overlaid subsampled data points."""

    plt.style.use("seaborn-v0_8-darkgrid")

    Y_full = np.array(Y_full)
    subsample_idx = np.array(subsample_idx)
    Y_subsampled = np.array(Y_subsampled)

    if Y_full.size != mesh_size**2:
        raise ValueError("Length of Y_full does not match mesh_size squared.")

    x = np.arange(mesh_size)
    y = np.arange(mesh_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z_full = Y_full.reshape(mesh_size, mesh_size)

    subsampled_y = subsample_idx // mesh_size
    subsampled_x = subsample_idx % mesh_size
    subsampled_z = Y_subsampled.flatten()

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_grid,
        Y_grid,
        Z_full,
        cmap=cmap,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.85,
        vmin=vmin,
        vmax=vmax,
        antialiased=True,
        rcount=100,
        ccount=100,
    )

    ax.scatter(
        subsampled_x,
        subsampled_y,
        subsampled_z,
        color="#FF6B6B",
        s=40,
        marker="o",
        alpha=0.9,
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )

    cbar = fig.colorbar(
        surf, shrink=cbar_shrink, aspect=cbar_aspect, pad=cbar_pad, ax=ax
    )
    cbar.set_label("GP Value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("X1", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_ylabel("X2", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_zlabel("Value", fontsize=11, fontweight="medium", labelpad=8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    ax.tick_params(axis="both", which="major", labelsize=9, pad=3)
    ax.view_init(elev=30, azim=-135)

    plt.tight_layout()
    plt.show()
