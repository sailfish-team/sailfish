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
  it often only requires changing a few lines of the kernel code.

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
To keep things simple, we will stick to two dimensions and we will build the
lid-driven cavity geometry, which is one of the standard testcases in
computational fluid dynamics.

The program outline
-------------------
In order to build a Sailfish simulation, we will create a new Python script.
In this script, we will need to import the ``lbm`` and ``geo`` Sailfish
modules::

    import lbm
    import geo

The ``lbm`` module contains a class which will drive our simulation, and the ``geo``
module contains classes used to describe the geometry of the simulaton.  We will start
with defining the main driver class for our example, and return to the issue of
geometry later.

Each Sailfish simulation is represented by a class derived from ``lbm.LBMSim``.
In the simplest case, we don't need to define any additional members of that class,
and a simple definition along the lines of::

    class LDCSim(lbm.LBMSim):
        pass

will do just fine.  The part of that class that is of primary interest to the end-user
is its ``__init__`` method.  When the class is instantiated, it parses the command
line arguments and stores the simulation settings in ``self.options`` (using the standard
Python ``optparse`` module).  The ``__init__`` method takes a single argument by default
-- the class representing the simulation geometry.

That class needs to be derived from either ``geo.LBMGeo2D`` or ``geo.LBMGeo3D``, depending
on the dimensionality of the problem at hand.  In our present case, we will
use the former one.  The derived geometry class needs to define at least the following
two methods: ``_define_nodes`` and ``init_dist``.

``_define_nodes`` is used to set the type of each node in the simulation domain.  The
size of the simulation domain is already known when the geometry class is instantiated
and can be accessed via its attributes ``lat_w`` (size along the X axis), ``lat_h``
(size along the Y axis) and ``lat_d`` (size along the Z axis, for 2D simulations always
equal to 1).

By default, the whole domain is initialized as fluid nodes.  To define the geometry, we
will want to redefine some of the nodes using the ``NODE_WALL``, ``NODE_VELOCITY`` or
``NODE_PRESSURE`` class constants.  ``NODE_WALL`` represents the no-slip condition at a
stationary domain boundary.  ``NODE_VELOCITY`` and ``NODE_PRESSURE`` represent a
boundary condition with specified velocity or pressure, respectively.  To redefine
the nodes, we will use the ``set_geo(location, type, data)`` function.  Here, ``location``
is a tuple representing the location of the node to update, ``type`` is one of the class
constants discussed above, and ``data`` is an optional argument used to specify the
imposed velocity or pressure.

In the lid-driven cavity (LDC) geometry, we consider a rectangular box, open at the top
where the fluid flows horizontally with some predefined velocity.  We therefore write
our function as follows::

    class LBMGeoLDC(geo.LBMGeo2D):
        max_v = 0.1
        def _define_nodes(self):
            for i in range(0, self.lat_w):
                self.set_geo((i, 0), self.NODE_WALL)
                self.set_geo((i, self.lat_h-1), self.NODE_VELOCITY, (self.max_v, 0.0))
            for i in range(0, self.lat_h):
                self.set_geo((0, i), self.NODE_WALL)
                self.set_geo((self.lat_w-1, i), self.NODE_WALL)

Now that we have the geometry out of the way, we can deal with the initial conditions.
This is done in the ``init_dist(dist)`` function, which is responsible for setting the initial
particle distributions in all nodes in the simulation domain.  The function takes a single
``dist`` argument, which is a numpy array containing the distributions.  We normally won't
be accessing that array directly anyway, so the exact details of how the distributions are
stored is irrelevant.  To set them, we will use the ``velocity_to_dist(location, velocity, dist)``
function, which will do all of the heavy lifting for us. To match our LDC geometry, we set
the velocity of the fluid everywhere to be 0, except for the first row at the top, where
we set the fluid to have to a ``max_v`` velocity in the horizontal direction::

        def init_dist(self, dist):
            self.velocity_to_dist((0,0), (0.0, 0.0), dist)
            self.fill_dist((0,0), dist)

            for i in range(0, self.lat_w):
                self.velocity_to_dist((i, self.lat_h-1), (self.max_v, 0.0), dist)

The only new thing here is the ``fill_dist`` function, which we use to copy the
distributions from node (0,0) to the whole simulation domain.  We do so to make the
code faster, as calculating the distributions multiple times and writing them to all
nodes one by one is a costly process, which is best avoided.

At this point, we are almost good to go.  The only remaining thing to do is to
instantiate the ``LDCSim`` class and use its ``run`` method to actually start the
simulation::

    sim = LDCSim(LBMGeoLDC)
    sim.run()

How it works behind the scenes
------------------------------

Using the command-line arguments
--------------------------------
TODO: Document the following options

* ``--lat_w``, ``--lat_h``, ``--lat_d``
* ``--precision``
* ``--benchmark``
* ``--backend``
* ``--batch``, ``--nobatch``
* ``--save_src``
* ``--use_src``
* ``--every``

Simulation results processing
=============================

Results form Sailfish simulations can saved to disk for processing in external
appliations or visualized in real-time.  These options are not exlusive, so
concurrent visualization and data saving is fully supported.

Data output
-----------

Sailfish supports two basic output data formats, provided the associated Python modules
are installed: HDF5 and VTK.  The data format is selected
using the ``--output_format`` command line option, which can take one of the
following values: ``h5nested``, ``h5flat``, ``vtk``.  The first two of those correspond
to different ways of saving the data to HDF5 files.  In all modes, the name of the
file is specified using the ``--output`` command line option, though the exact meaning
of this is different for different file formats.

In the ``h5nested`` mode, a single HDF5 file is created.  The file contains a single
group called ``results``.  The group contains a single table and each record in this
table corresponds to the state of the simulation at a specific iteration.  The table
contains columns for velocity components (``vx``, ``vy`` and ``vz``), density (``rho``)
and the current iteration (``iter``).

In the ``h5flat`` mode, also a single HDF5 file is created.  The file contains
multiple groups, called ``iterXXX`` where ``XXX`` is the iteration number.  Each
of the groups contains two arrays: ``v`` for the velocity vector field, and ``rho``
for the density scalar field.

In the ``vtk`` mode, multiple XML VTK files are generated, each containing the state
of the simulation at a specific iteration.  The file names are generated by appending
the iteration number to the base name provided via the ``--output`` option, e.g. if
``--output=poiseuille`` is used, ``poiseuille00400.xml`` will contain data for the
400-th iteration.  The output files contain two fields: a scalar field named ``density``
and a vector field called ``velocity``.

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

The initial size of the visualization window can be controlled with command line parameters.
One can specify its size explicitly using the ``--scr_w`` and ``--scr_h`` options
to set the width and height, respectively.  Alternatively, the ``--scr_scale`` option can
be used to make the window dimensions directly proportional to those of the lattice, e.g.
if ``--scr_scale=3`` is used the window will be 3 times larger than the lattice and each
lattice site will be visualized as a 3x3 square.

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
* i: toggle visibility of text info about the state of the simulation (MLUPS, iteration etc.)
* v: toggle visualization of the velocity vector field
* t: toggle visualization of the fluid tracer particles
* c: toggle convolution of the visualization with a Gaussian kernel (this has a smoothing effect)
* r: reset the simulation geometry (this clears any obstacles added interactively)
* q: quit the simulation
* s: take a screenshot
* ,: decrease the max value used for visualization
* .: increase the max value used for visualization

The *max value* above corresponds to either the maximum velocity or maximum vorticity,
depending on the current visualization mode.  Changing this value will affect the color
scale of the visualized field.

Visualization of 3D data
^^^^^^^^^^^^^^^^^^^^^^^^

3D data visualization is provided via the mayavi package.  This visualization is
not interactive at this time.

Supported models
================

The Sailfish solver currently supports the following Lattice-Boltzmann models and grids:

* two-dimensional: D2Q9 (BGK, MRT models)
* three-dimensional: D3Q13 (BGK), D3Q15, D3Q19 (BGK, MRT)

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

Lid-driven cavity (2D)
----------------------
.. image:: img/ldc2d.png

The image illustrates fluid velocity visualized in the ``rgb1`` mode.  The cyan circles are
tracer particles.  The red lines depict the velocity field.

Poiseuille flow (2D)
--------------------

Flow around a cylinder (2D)
---------------------------

.. image:: img/cylinder2d_vorticity.png

Von Kármán vortex street.  The image illustrates vorticity visualized in the ``2col`` mode.

Flow around a sphere (3D)
-------------------------

.. image:: img/sphere3d.png
