Supported models, grids and boundary conditions
===============================================

The Sailfish solver currently supports the following Lattice-Boltzmann models and grids:

* single fluid

  * two-dimensional: D2Q9 (BGK, MRT, entropic models)
  * three-dimensional: D3Q13 (MRT), D3Q15 (entropic), D3Q19 (BGK, MRT models)

* binary fluid

  * two-dimensional: Shan-Chen (D2Q9), free-energy (D2Q9, BGK, MRT) [PRE78]_
  * three-dimensional: Shan-Chen, free-energy (D3Q19, BGK, MRT)

.. [PRE78] Contact line dynamics in binary lattice Boltzmann simulations, Phys. Rev. E 78, 056709 (2008). DOI: 10.1103/PhysRevE.78.056709

The single fluid models are implemented in both the incompressible and weakly compressible version, the
latter of which is the default.  To turn on the incompressible model, use the ``--incompressible``
command line switch.

An external force field (body force) can be enabled in all models.

A general overview of the Lattice-Boltzmann method
--------------------------------------------------

The Lattice-Boltzmann method is based on solving the following equation:

.. math:: f_\alpha(\vec{x_i} + \vec{e_\alpha}, t + 1) - f_\alpha(\vec{x_i}, t) = \Omega_i

which is a discrete version of the Boltzmann equation known from non-equilibrium
statistical mechanics.  Here, :math:`f_\alpha` are distributions of particles
moving in the direction :math:`\vec{e_\alpha}` (the available directions are specified
by the chosen grid), :math:`t` is the current time step, :math:`\vec{x_i}` is the
position of the :math:`i`-th node in the grid, and :math:`\Omega_i` is the collision
operator.

TODO: Add more info about the LBM.

Single relaxation time (BGK)
----------------------------

The BGK (Bhatnagar-Gross-Krook) approximation is based on the following form
of the collision operator:

.. math:: \Omega_i = -\frac{f(\vec{x_i}, t) - f^{(eq)}(\rho_i, \vec{v_i})}{\tau}

where :math:`\tau` is the relaxation time and :math:`f^{(eq)}` is the equilibrium
distribution, defined as a function of macroscopic variables at a node.

TODO: Add more info about the BGK approximation and its limitations.

Multiple relaxation times (MRT)
-------------------------------

TODO: Add info about the MRT model.

Boundary conditions
-------------------
Check the :mod:`node_type` module for a full and up-to-date list of supported boundary
conditions.
