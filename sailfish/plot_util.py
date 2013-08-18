"""Util functions for plots using internal Sailfish data structures."""

from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations, product
import numpy as np

class Arrow3D(FancyArrowPatch):
    """Simple 2D arrow plotted in 3D space."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _plot_grid3d(ax, grid, bbox, planes):
    """Plots a 3D LB lattice.

    :param ax: matplotlib Axes object to use for plotting
    :param grid: Sailfish grid object to illustrate
    :param bbox: if True, draw a blue 3D bounding box around the plot
    :param planes: if True, draws planes passing through the origin
    """
    assert grid.dim == 3
    bb = 0
    for ei in grid.basis[1:]:
        a = Arrow3D(*zip((0, 0, 0), [float(x) * 1.05 for x in ei]), color='k', arrowstyle='-|>',
                    mutation_scale=10, lw=1)
        ax.add_artist(a)
        bb = max(bb, max([int(x) for x in ei]))

    for i in ('x', 'y', 'z'):
        c = Circle((0, 0), radius=0.05, color='k')
        ax.add_patch(c)
        art3d.pathpatch_2d_to_3d(c, z=0, zdir=i)

    ax.set_xlim(-bb, bb)
    ax.set_ylim(-bb, bb)
    ax.set_zlim(-bb, bb)

    if planes:
        p1 = [(-bb, -bb, 0), (bb, -bb, 0), (bb, bb, 0), (-bb, bb, 0)]
        p2 = [(0, -bb, -bb), (0, bb, -bb), (0, bb, bb), (0, -bb, bb)]
        p3 = [(-bb, 0, -bb), (bb, 0, -bb), (bb, 0, bb), (-bb, 0, bb)]

        ax.add_collection3d(Poly3DCollection([p1, p2, p3], facecolor='b', lw=0,
                                             alpha=0.1))

    if bbox:
        r = [-bb, bb]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s - e)) == r[1] - r[0]:
                ax.plot3D(*zip(s, e), color='b', ls='--')


def _plot_grid2d(ax, grid):
    """Plots a 2D LB lattice.

    :param ax: matplotlib Axes object to use for plotting
    :param grid: Sailfish grid object to illustrate
    """
    assert grid.dim == 2
    bb = 0
    for ei in grid.basis[1:]:
        a = FancyArrowPatch((0, 0), [float(x) * 1.025 for x in ei], color='k',
                            arrowstyle='-|>', lw=1, mutation_scale=10.0)
        #mutation_scale=3)
        ax.add_artist(a)
        bb = max(bb, max([int(x) for x in ei]))

    a = Circle((0, 0), radius=0.05, color='k')
    ax.add_artist(a)

    ax.set_xlim(-bb, bb)
    ax.set_ylim(-bb, bb)

def plot_grid(ax, grid, bbox=False, planes=False):
    """Plots a LB lattice.

    :param ax: matplotlib Axes object to use for plotting
    :param grid: Sailfish grid object to illustrate
    :param bbox: if True, draw a blue 3D bounding box around the plot
    :param planes: if True, draws planes passing through the origin
    """
    if grid.dim == 3:
        return _plot_grid3d(ax, grid, bbox, planes)
    else:
        return _plot_grid2d(ax, grid)
