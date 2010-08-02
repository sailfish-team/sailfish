Running simulations
===================

In this section, we show how to create a simple Lattice-Boltzmann simulation using Sailfish.
To keep things simple, we stick to two dimensions and use the lid-driven cavity
example, which is one of the standard test cases in computational fluid dynamics.

The program outline
-------------------
In order to build a Sailfish simulation, we create a new Python script.
In this script, we need to import the :mod:`lbm` and :mod:`geo` Sailfish
modules::

    from sailfish import lbm, geo

The :mod:`lbm` module contains a class which will drive our simulation, and the :mod:`geo`
module contains classes used to describe the geometry of the simulation.  We will start
by defining the main driver class for our example, and return to the issue of
geometry later.

Each Sailfish simulation is represented by a class derived from :class:`lbm.FluidLBMSim`.
In the simplest case, we don't need to define any additional members of that class,
and a simple definition along the lines of::

    class LDCSim(lbm.FluidLBMSim):
        pass

will do just fine.  The part of that class that is of primary interest to the end-user
is its ``__init__`` method.  When the class is instantiated, it parses the command
line arguments and stores the simulation settings in ``self.options`` (using the standard
Python :py:mod:`optparse` module).  The ``__init__`` method takes a single argument by default
-- the class representing the simulation geometry.

That class needs to be derived from either :class:`geo.LBMGeo2D` or :class:`geo.LBMGeo3D`, depending
on the dimensionality of the problem at hand.  In our present case, we will
use the former one.  The derived geometry class needs to define at least the following
two methods: ``define_nodes`` and ``init_dist``.

``define_nodes`` is used to set the type of each node in the simulation domain.  The
size of the simulation domain is already known when the geometry class is instantiated
and can be accessed via its attributes ``lat_nx`` (size along the X axis), ``lat_ny``
(size along the Y axis) and ``lat_nz`` (size along the Z axis, for 2D simulations always
equal to 1).

By default, the whole domain is initialized as fluid nodes.  To define the geometry, we
need to redefine some of the nodes using the :const:`geo.LBMGeo.NODE_WALL`, :const:`geo.LBMGeo.NODE_VELOCITY` or
:const:`geo.LBMGeo.NODE_PRESSURE` class constants.  :const:`geo.LBMGeo.NODE_WALL` represents a no-slip condition at a
stationary domain boundary.  :const:`geo.LBMGeo.NODE_VELOCITY` and :const:`geo.LBMGeo.NODE_PRESSURE` represent a
boundary condition with specified velocity or pressure, respectively.  To redefine
the nodes, we will use the ``set_geo(location, type, data)`` function.  Here, ``location``
is either a tuple representing the location of the node to update, or a NumPy Boolean
array.  Using NumPy arrays is preferred, as they are much faster for larger domains.
As for the remaining arguments of ``set_geo``, ``type`` is one of the class constants
discussed above, and ``data`` is an optional argument used to specify the imposed
velocity or pressure.

In the lid-driven cavity (LDC) geometry, we consider a rectangular box, open at the top
where the fluid flows horizontally with some predefined velocity.  We therefore write
our function as follows::

    class LBMGeoLDC(geo.LBMGeo2D):
        max_v = 0.1

        def define_nodes(self):
            hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]
            wall_map = np.logical_or(
                    np.logical_or(hx == self.lat_nx-1, hx == 0), hy == 0)

            self.set_geo(hy == self.lat_ny-1, self.NODE_VELOCITY, (self.max_v, 0.0))
            self.set_geo(wall_map, self.NODE_WALL)

Now that we have the geometry out of the way, we can deal with the initial conditions.
This is done in the ``init_dist(dist)`` function, which is responsible for setting the initial
particle distributions in all nodes in the simulation domain.  The function takes a single
``dist`` argument, which is a NumPy array containing the distributions.  We normally won't
be accessing that array directly anyway, so the exact details of how the distributions are
stored is irrelevant.  

There are two ways to set their initial value.  The first one is based on the
``velocity_to_dist(location, velocity, dist)`` function, which sets the node at ``location``
to have the equilibrium distribution corresponding to ``velocity`` and a density of 1.0.
The alternative way of specifying initial conditions is to provide the values of macroscopic
variables (density, velocity) everywhere in the simulation domain, and let the GPU calculate
the equilibrium distributions.  The second method is preferred, as it is faster and requires
less memory on the host.

In our LDC geometry, we set the velocity of the fluid everywhere to be 0 (this is the default value
so we do not have to specify this explicitly), except for the first row at the top, where we set
the fluid to have to a ``max_v`` velocity in the horizontal direction::

        def init_dist(self, dist):
            hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

            self.sim.ic_fields = True
            self.sim.rho[:] = 1.0
            self.sim.vx[hy == self.lat_ny-1] = self.max_v

At this point, we are almost good to go.  The only remaining thing to do is to
instantiate the ``LDCSim`` class and use its ``run`` method to actually start the
simulation::

    sim = LDCSim(LBMGeoLDC)
    sim.run()

How it works behind the scenes
------------------------------
When the :func:`lbm.LBMSim.run` method is called, Sailfish instantiates the geometry class (this
process can take a few seconds for 3D simulations with complex ``init_dist()`` and
``define_nodes()`` functions.  It then uses the Mako template engine and the information
from the options and the geometry class to generate the code for the compute
unit (e.g. a GPU).  The code can be in either CUDA C or OpenCL and it is
automatically optimized (e.g. code for models and boundary conditions other than the
selected ones is automatically removed).  The generated code is then compiled on the
fly by the :mod:`pyopencl` or :mod:`pycuda` modules into a binary which is executed on the GPU.

The template for the compute unit source is contained in the ``.mako`` files in the ``templates``
directory of the :mod:`sailfish` module.  It is written in a mix of Python, Mako and CUDA C.  
Parts of the code that end up in GPU functions are also generated by the :mod:`sym` module.  
This module contains functions which return SymPy expressions, which are then converted to C code.
The use of :mod:`sympy` makes it possible to write large parts of the code in a grid-independent form, which
is then automatically expanded when the GPU code is generated.

This process, although seemingly quite complex, has several advantages:

* The generated code can be automatically optimized.
* The code for multiple targets can be generated automatically (currently, OpenCL and
  CUDA are supported).
* By keeping the source code in a grid-independent form, the code becomes easier to
  read and can work automatically with new grids and models.

Using the command-line arguments
--------------------------------
The base class for Sailfish simulations (:class:`lbm.LBMSim`) defines a large number of command line
options which can be used to control the simulation.  To get a full list of currently supported
options, run any Sailfish simulation with the ``--help`` command line option.  Some of the
basic settings you might want to play with when starting to work with Sailfish are as follows:

* ``--lat_nx=N``, ``--lat_ny=N``, ``--lat_nz=N``: set lattice dimensions (width, height and depth, respectively)
* ``--precision=X``: set the precision of floating-point numbers used in the simulation (``single`` or ``double``).
  Note that double precision calculations will currently be significantly slower than their single precision
  counterparts, and might not be supported at all on some older devices.
* ``--backend=X``: select the backend to be used to run the simulation.  Supported values are
  ``cuda`` and ``opencl``.  Their availability will depend on the presence of required Python
  modules in the host system (:mod:`pyopencl`, :mod:`pycuda`).
* ``--save_src=FILE``: save the generated GPU code to ``FILE``.
* ``--use_src=FILE``: use the GPU code from ``FILE`` instead of the one generated by Sailfish
  (useful for testing minor changes in the kernel code).
* ``--every=N``: update the display every ``N`` iterations.
* ``--benchmark``: run the simulation in benchmark mode, printing information about its
  performance to the standard output.
* ``--batch``, ``--nobatch``: force or disable batch mode, respectively.  In batch mode, all
  visualization modules are disabled and hooks defined for the simulation are run at
  specified iterations.  Batch mode requires specifying the ``max_iters`` option.
* ``--max_iters=N``: the number of iterations the simulation is to be run for in batch mode.

The ``--save_src`` option is particularly useful if you want to learn the basic structure of the
GPU code.  The Mako template files, which contain the actual code, can be difficult to
understand at first, as they mix three languages: Python, the Mako template language and
CUDA C.  To avoid this complexity, you might want to save the generated compute device code
and inspect it in a text editor.  The generated code will be automatically formatted to be
readable unless the ``--noformat_src`` option is specified.  The command used to format the
code can be redefined by overriding the :attr:`lbm.LBMSim.format_cmd` value.  The default one
requires the ``indent`` utility and is set so that the generated code roughly follows the
formatting style of the Linux kernel (with longer lines, which can be useful for complex expressions).

Troubleshooting
---------------

My simulation works fine in single precision, but breaks in double precision.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If your simulation runs in double precision, but generates clearly unphysical results that
do not appear when it's run in single precision, it's possible that the CUDA optimizing compiler
is generating broken code.  To check whether this is the case, you need to disable all optimizations
by running your simulation with the ``--cuda-nvcc-opts="-Xopencc -O0"`` command line option.
Note that this will significantly decrease the performance of your simulation.

