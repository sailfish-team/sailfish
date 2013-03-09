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
    from sailfish.subdomain import Subdomain2D
    from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
    from sailfish.controller import LBSimulationController
    from sailfish.lb_single import LBFluidSim

    class LDCBlock(Subdomain2D):
        max_v = 0.1

        def boundary_conditions(self, hx, hy):
            wall_map = (((hx == self.gx-1) | (hx == 0) | (hy == 0)) &
                        np.logical_not(hy == self.gy-1))
            self.set_node(hy == self.gy-1, NTEquilibriumVelocity((self.max_v, 0.0)))
            self.set_node(wall_map, NTFullBBWall)

        def initial_conditions(self, sim, hx, hy):
            sim.rho[:] = 1.0
            sim.vx[hy == self.gy-1] = self.max_v


    class LDCSim(LBFluidSim):
        subdomain = LDCBlock

    if __name__ == '__main__':
        ctrl = LBSimulationController(LDCSim).run()


Want to see Sailfish in action?  Check out our `videos on YouTube <http://www.youtube.com/watch?v=kx4-VjaJ2eI&feature=PlayList&p=96C9241314F1A898&index=0&playnext=1>`_,
or better yet, `get the code <http://github.com/sailfish-team>`_ and see for yourself by running the provided examples.

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
   api
   about
   release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

