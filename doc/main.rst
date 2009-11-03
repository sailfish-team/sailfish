Motivation and Design Principles
================================

Sailfish is a general purpose fluid dynamics solver optimized for modern multicore processors,
especially Graphics Processing Units (GPUs).  The solver is based on the Lattice Boltzmann Method,
which is conceptually quite simple to understand and which scales very well with increasing
computational resources.

The Sailfish project is also an experiment in scientific computing and software engineering.
Unlike the majority of CFD packages, which are written in compiled languages such as C++
or Fortran, Sailfish is implemented in Python and CUDA C/OpenCL.  We have found this
combination to be a very powerful one, making it possible to significantly shorten
development time without sacrificing any computational performance.

The general goals of the project are as follows:

* **performance** the code is optimized for the current generation of NVIDIA GPUs.
  With a single state of the art video board, it is possible to achieve a simulation speed
  of about 700 MLUPS.  To achieve comparable performance with typical off-the-shelf CPUs,
  a small cluster would be necessary.

* **scalability**: the code is designed to scale well (i.e. linearly) with
  increasing number of compute cores.

* **agility and extendability**: by implementing large parts of the code in a very
  expressive language (Python), we aim at encouraging rapid experimentation.
  Running tests, playing with new boundary conditions or new models is easy, as
  often only requires changing a few lines of the kernel code.

* **maintainability**: we keep the code clean and easy to understand.  The Mako
  templating engine makes it possible to dynamically generate optimized code without
  any unnecessary cruft.

* **ease of use**: defining new simulations and exploring simulation results is
  simple and many details are automated and hidden from the end-user.

Installation
============

Sailfish requires no installation and all sample simulations provided in the executable
.py files can simply be started from a shell, provided the required packages are
installed on the host system.  These are as follows:

General requirements:

* numpy
* sympy-0.6.5
* mako-0.2.5
* a Python module for the computational backend (one or more of the following):

  * pycuda-0.92 (with the NVIDIA drivers and NVIDIA CUDA Toolkit)
  * pypencl (with any OpenCL implementation)

Visualization:

* pygame (for 2D)
* mayavi (for 3D)

Data output:

* pytables-2.1.1 (HDF5 output)
* tvtk (VTK output)

Tests:

* matplotlib

Tutorial
========

In this section, we show how to create a simple LBM simulation using Sailfish.
We will stick to two dimension and we will build the lid-driven cavity geometry,
which is one of the standard testcases in computational fluid dynamics.

TODO

Simulation results processing
=============================

TODO

Data output
-----------

TODO

Data visualization
------------------

Sailfish supports on-line data visualization without writing out any results
to files on disk.  The visualization modules for 2D and 3D are completely different,
and thus they will be discussed separately.

Visualization of 2D data
^^^^^^^^^^^^^^^^^^^^^^^^

2D simulations can be monitored using an interactive pygame-based interface.
The interface supports the following color schemes:

* ``std`` (default): a simple palette with a single color (yellow)
* ``rgb1``: default color palette from gnuplot
* ``hsv``: HSV colorspace; visualized quantities determine the 'hue' component
* ``2col``: a scheme with 2 colors: blue (for negative values) and red (for positive ones)

The color schemes can be selected from the command line using the ``--vismode`` option.

Interaction with the simulation is provided via mouse: left key presses place walls
(i.e. nodes with the no-slip boundary condition, with no intrinsic velocity) in the
simulation domain, and right key presses remove them.

The visualization module can be controlled from the keyboard, and the following
keys are defined:

* 0: visualize the norm of the fluid velocity
* 1: visualize the x component of the fluid velocity
* 2: visualize the y component of the fluid velocity
* 3: visualize variations in the fluid density
* 4: visualize the vorticity of the fluid
* v: toggle visualization of the velocity vector field
* t: toggle visualization of the fluid tracer particles
* c: toggle convolution of the visualization with a Gaussian kernel (this has a smoothing effect)
* r: reset the simulation geometry (this clears any obstacles added interactively)
* q: quit the simulation
* s: take a screenshot

Visualization of 3D data
^^^^^^^^^^^^^^^^^^^^^^^^

3D data visualization is provided via the mayavi package.  This visualization is
not interactive at this time.

Supported models
================

The Sailfish solver currently supports the following Lattice-Boltzmann models and grids:

* two-dimensional: D2Q9 (BGK, MRT models)
* three-dimensional: D3Q13, D3Q15, D3Q19 (the BGK model)

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

Zou-He
------

Test cases
==========

Lid-driven cavity
-----------------
.. image:: img/ldc2d.png

The image illustrates fluid velocity visualized in the ``rgb1`` mode.  The cyan circles are
tracer particles.  The red lines depict the velocity field.

Poiseuille flow
---------------

Flow around a cylinder
----------------------

.. image:: img/cylinder2d_vorticity.png

Von Kármán vortex street.  The image illustrates vorticity visualized in the ``2col`` mode.


