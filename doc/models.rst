Supported models, grids and boundary conditions
===============================================

The Sailfish solver currently supports the following Lattice-Boltzmann models and grids:

* two-dimensional: D2Q9 (BGK, MRT models)
* three-dimensional: D3Q13 (MRT), D3Q15, D3Q19 (BGK, MRT models)

The Lattice-Boltzmann method is based on solving the following equation:

.. math:: f_\alpha(\vec{x_i} + \vec{e_\alpha}, t + 1) - f_\alpha(\vec{x_i}, t) = \Omega_i

which is a discrete version of the Boltzmann equation known from non-equilibrium
statistical mechanics.  Here, :math:`f_\alpha` are distributions of particles
moving in the direction :math:`\vec{e_\alpha}` (the available directions are specified
by the chosen grid), :math:`t` is the current time step, :math:`\vec{x_i}` is the
position of the :math:`i`-th node in the grid, and :math:`\Omega_i` is the collision
operator.

TODO

Single relaxation time (BGK)
----------------------------

The BGK (Bhatnagar-Gross-Krook) approximation is based on the following form
of the collision operator:

.. math:: \Omega_i = -\frac{f(\vec{x_i}, t) - f^{(eq)}(\rho_i, \vec{v_i})}{\tau}

where :math:`\tau` is the relaxation time and :math:`f^{(eq)}` is the equilibrium
distribution, defined as a function of macroscopic variables at a node.

TODO

Multiple relaxation times (MRT)
-------------------------------

TODO

Boundary conditions
===================

Bounce-back
-----------
TODO

Equilibrium
-----------
TODO

Zou-He
------
TODO



