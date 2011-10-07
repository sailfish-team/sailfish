.. Sailfish documentation master file, created by
   sphinx-quickstart on Tue Nov  3 22:53:06 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#########################
Sailfish Reference Manual
#########################

:Release: |version|
:Date: |today|

.. module:: sailfish

Sailfish is a free computational fluid dynamics solver based on the Lattice Boltzmann
method and optimized for modern multi-core systems, especially GPUs (Graphics Processing Units).

To illustrate how easy it is to create simulations using the Sailfish package,
here is a simple example code to simulate fluid flow in a lid-driven cavity::

    import numpy as np
    from sailfish import geo, lbm

    class LBMGeoLDC(geo.LBMGeo2D):
        max_v = 0.1

        def define_nodes(self):
            hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]
            wall_map = np.logical_or(
                    np.logical_or(hx == self.lat_nx-1, hx == 0), hy == 0)

            self.set_geo(hy == self.lat_ny-1, self.NODE_VELOCITY, (self.max_v, 0.0))
            self.set_geo(wall_map, self.NODE_WALL)

        def init_dist(self, dist):
            hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

            self.sim.ic_fields = True
            self.sim.rho[:] = 1.0
            self.sim.vx[hy == self.lat_ny-1] = self.max_v

    class LDCSim(lbm.FluidLBMSim):
        pass

    if __name__ == '__main__':
        LDCSim(LBMGeoLDC).run()

Want to see Sailfish in action?  Check out our `videos on YouTube <http://www.youtube.com/watch?v=kx4-VjaJ2eI&feature=PlayList&p=96C9241314F1A898&index=0&playnext=1>`_,
or better yet, `get the code <http://gitorious.org/sailfish>`_ and see for yourself by running the provided examples.

.. youtube:: kx4-VjaJ2eI

Contents
========

.. toctree::
   :maxdepth: 3

   intro
   installation
   tutorial
   models
   performance
   internals
   testcases
   regtest
   api
   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

