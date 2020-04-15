from typing import List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_tensor(tensors: List[np.array], descriptions: List, cmap=cm.jet):
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=0, vmax=len(tensors))
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for idx, tensor in enumerate(tensors):

        eig_vals, eig_vecs = np.linalg.eigh(tensor)

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = eig_vals[0] * np.outer(np.cos(u), np.sin(v))
        y = eig_vals[1] * np.outer(np.sin(u), np.sin(v))
        z = eig_vals[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eig_vecs.T)

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(idx), linewidth=0.5, alpha=0.3, shade=True)

    legend_elements = [Line2D([0], [0], marker='o', color=m.to_rgba(idx), label=description,
                              markerfacecolor=m.to_rgba(idx), markersize=20) for idx, description in enumerate(descriptions)]
    ax.legend(handles=legend_elements, prop={'size': 20})