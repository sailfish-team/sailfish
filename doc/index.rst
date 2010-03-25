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

Sailfish is a free computational fluid dynamics solver employing the Lattice Boltzmann
method and optimized for modern commodity high-performance computational systems,
especially Graphics Processing Units.

To illustrate how easy it is to create simulations using the Sailfish package,
here is a simple example code to simulate fluid flow in a lid-driven cavity::

    from sailfish import lbm
    from sailfish import geo

    class LBMGeoLDC(geo.LBMGeo2D):
        max_v = 0.1

        def define_nodes(self):
            for i in range(0, self.lat_nx):
                self.set_geo((i, 0), self.NODE_WALL)
                self.set_geo((i, self.lat_ny-1), self.NODE_VELOCITY, (self.max_v, 0.0))
            for i in range(0, self.lat_ny):
                self.set_geo((0, i), self.NODE_WALL)
                self.set_geo((self.lat_nx-1, i), self.NODE_WALL)

        def init_dist(self, dist):
            self.velocity_to_dist((0,0), (0.0, 0.0), dist)
            self.fill_dist((0,0), dist)

            for i in range(0, self.lat_nx):
                self.velocity_to_dist((i, self.lat_ny-1), (self.max_v, 0.0), dist)

    class LDCSim(lbm.FluidLBMSim):
        pass

    sim = LDCSim(LBMGeoLDC)
    sim.run()

Want to see Sailfish in action?  Check out our `videos on YouTube <http://www.youtube.com/watch?v=kx4-VjaJ2eI&feature=PlayList&p=96C9241314F1A898&index=0&playnext=1>`_,
or better yet, `get the code <http://gitorious.org/sailfish>`_ and see for yourself by running the provided examples.

.. youtube:: kx4-VjaJ2eI

Contents
========

.. toctree::
   :maxdepth: 2

   intro
   installation
   tutorial
   simulations
   results
   models
   testcases
   regtest
   api
   about

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

