import h5py
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
import matplotlib.gridspec as gridspec


# Open file
hdf = h5py.File("DustModelI.h5", "r")

# Define colors
cols = [("Webb.NIRCAM.F150W", "Webb.NIRCAM.F200W"),
        ("Webb.NIRCAM.F200W", "Webb.NIRCAM.F277W"),
        ("Webb.NIRCAM.F277W", "Webb.NIRCAM.F356W"),
        ("Webb.NIRCAM.F356W", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F410M", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F430M", "Webb.NIRCAM.F444W"),
        ("Webb.NIRCAM.F444W", "Webb.MIRI.F560W"),
        ("Webb.MIRI.F560W", "Webb.MIRI.F770W")]

# Load redshifts
gal_zs = hdf["z"][...]
zs = np.arange(4.5, 15.6, 0.1)

print(zs)

# Define plot
ncols = len(cols)
fig = plt.figure(figsize=(ncols * 3.5, 3.75))
gs = gridspec.GridSpec(nrows=2, ncols=ncols,
                       height_ratios=[1, 20])
gs.update(wspace=0.0, hspace=0.0)
axes = []
caxes = []
for i in range(ncols):
    axes.append(fig.add_subplot(gs[1, i]))
    caxes.append(fig.add_subplot(gs[0, i]))

# Define plotting parameters
extent = [np.min(zs), np.max(zs), np.min(zs), np.max(zs)]
norm = LogNorm(vmin=1, vmax=1)

# Loop over colors
for (i, ax), cax, (c1, c2) in zip(enumerate(axes), caxes, cols):

    print(c1, "-", c2)

    # Get fluxes
    f1 = hdf[c1][...]
    f2 = hdf[c2][...]

    # Get color
    col = 2.5 * np.log10(f1 / f2)

    bin_col, _, _ = binned_statistic(gal_zs, col, statistic="median", bins=zs)

    # Compute grid
    XX, YY = np.meshgrid(bin_col, bin_col)

    # Compute residual
    resi = np.abs(XX - YY)

    print(np.min(resi), np.max(resi))

    # Plot heat map
    im = ax.imshow(resi, extent=extent, cmap="magma", norm=norm)

    cbar = fig.colorbar(im, cax, orientation="horizontal")
    cbar.set_label(r"$|A-B(z_{x}) - A-B(z_{y})|$")

    ax.set_xlabel("$z$")
    if i == 0:
        ax.set_ylabel("$z$")
    if i > 0:
        ax.tick_params("y", left=False, right=False, labelleft=False,
                       labelright=False)

fig.savefig("color_confusion.png", bbox_inches="tight", dpi=300)
