Supported models, grids and boundary conditions
===============================================

The Sailfish solver currently supports the following Lattice-Boltzmann models and grids:

* two-dimensional: D2Q9 (BGK, MRT models)
* three-dimensional: D3Q13 (MRT), D3Q15, D3Q19 (BGK, MRT models)

The models are implemented in both the incompressible and weakly compressible version, the
latter of which is the default.  To turn on the incompressible model, use the ``--incompressible``
command line switch.

An external force field (body force) can be enabled in all models.  The force field
is defined as accelerations using the ``--accel_x``, ``--accel_y``, and ``--accel_z``
parameters.  The field is uniform and acts globally in the whole simulation domain.

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

Sailfish supports a number of local boundary conditions (non-local boundary conditions are
currently not supported).  Each of the implemented boundary condition types can be available
for a specific kind(s) of boundary conditions (no-slip (wall), velocity, pressure) and dimensions.

Bounce-back
-----------
This is the simplest kind of boundary condition, available for no-slip and velocity nodes in
all dimensions.  The bounce-back algorithm works by reflecting all distributions across the
node center, and optionally adjusting some of the distributions to impose a velocity boundary
condition.

Equilibrium
-----------
This boundary condition algorithm works by imposing the BGK equilibrium distritbutions for
a node.  It currently works for all types of boundary conditions in two dimensions only.

Zou-He
------
This algorithm is based upon the idea of the reflection of the off-equilibrium distributions.
It currently works for all types of boundary conditions in two dimensions only.


