Running simulations
===================

In this section, we show how to create a simple lattice Boltzmann simulation
using Sailfish.
To keep things simple, we stick to two dimensions and use the lid-driven cavity
example, which is one of the standard test cases in computational fluid
dynamics. Here is the complete example which we are going to analyze step by step::

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
        LBSimulationController(LDCSim).run()


The program outline
-------------------
In order to build a Sailfish simulation, we create a new Python script. In this
script, we need to import the :mod:`lb_single`, :mod:`controller`, :mod:`subdomain`
Sailfish modules::

    import numpy as np
    from sailfish.subdomain import Subdomain2D
    from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
    from sailfish.controller import LBSimulationController
    from sailfish.lb_single import LBFluidSim

The :mod:`controller` module contains a class which will drive our simulation.
The :mod:`subdomain` module contains classes used to describe the geometry of the
simulation and is used to define the boundary and initial conditions.

Each single fluid Sailfish simulation is represented by a class derived
from :class:`lb_single.LBFluidSim`.
In the simplest case, we just have to define a subdomain class::

    class LDCSim(LBFluidSim):
        subdomain = LDCBlock

:class:`LDCBlock` is a required class, derived from :class:`subdomain.Subdomain2D`
or :class:`subdomainSubdomain3D`, depending on the
dimensionality of the problem at hand. In our present case, we will use the
former one. This class represents the geometry of the simulation.

In :class:`LDCSim` we can define the default parameter values and
additional command line arguments. To change the default parameters we have to
create the class method ``update_defaults`` where we update the ``defaults``
dictionary with the desired values of selected parameters. In our case we have to
change the size along the X axis (``lat_nx``) and the size along the Y axis
(``lat_ny``). This method is called before the class is instantiated::

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})

To add additional command line arguments we will create the class method
``add_options``. This method takes two arguments. ``group`` is a group of settings
connected with running the simulation, and ``dim`` is the dimension of the simulation
domain. This method, like ``update_defaults``, is called before the class is
instantiated. When the simulation is running, the command line arguments are
parsed and their settings are stored in ``self.config`` (using the standard
Python :py:mod:`argparse` module). In the first place, this method calls the same
methods in superclasses. After that we can add our options::

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        group.add_argument('--subdomains', type=int, default=1, help='number of subdomains to use')

:class:`LDCBlock` describes the simulation geometry and inherits from
:class:`Subdomain2D`. The derived geometry class needs to define at least the
following two methods: ``boundary_conditions`` and ``initial_conditions``.

``boundary_conditions`` is used to set the type of each node in the simulation
domain. The function takes two arguments: ``hx`` and ``hy``, which are NumPy
arrays constructed using the mgrid mechanism. We normally won’t be accessing these
parameters directly anyway, so the exact details of how the distributions are
stored are irrelevant at this point. The size of the simulation domain is already
known when the geometry class is instantiated and can be accessed via its
attributes ``gx`` (size along the X axis) and ``gy`` (size along the Y axis).

By default, the whole domain is initialized as fluid nodes. To define the
geometry, we need to redefine some of the nodes using the
:class:`node_type.NTFullBBWall` or :class:`node_type.NTEquilibriumVelocity`
classes to set a no-slip condition or enfore a constant fluid velocity, respectively.

To redefine the nodes, we will use the ``set_node(location, node_type)`` function.
``location`` is a NumPy Boolean array and ``node_type`` is a class object or a class
instance identifying the type of the boundary condition.  If the condition does
not take any parameter, it's enough to provide a class object.  Otherwise, an
instance needs to be created by providing the necessary parameters to the class
constructor.

In the lid-driven cavity (LDC) geometry, we consider a rectangular box, open at
the top where the fluid flows horizontally with some predefined velocity. We
therefore write our function as follows::

    class LDCBlock(Subdomain2D):
        max_v = 0.1

        def boundary_conditions(self, hx, hy):
            wall_map = (((hx == self.gx-1) | (hx == 0) | (hy == 0)) &
                        np.logical_not(hy == self.gy-1))
            self.set_node(hy == self.gy - 1, NTEquilibriumVelocity((self.max_v, 0.0)))
            self.set_node(wall_map, NTFullBBWall)

Note that by using Boolean operations on NumPy arrays we took care to make sure
that the velocity and wall nodes do not overlap.  This is intentional as
redefining node types is not allowed in Sailfish.

Now that we have the geometry out of the way, we can deal with the initial
conditions. This is done in the ``initial_conditions`` function, which takes
three arguments: ``hx``, ``hy`` and ``sim``, where ``sim`` is the simulation object.

The way of specifying initial conditions is to provide the values of macroscopic
variables (density, velocity) everywhere in the simulation domain, and let the
GPU calculate the particle distributions using the equilibrium function.

In our LDC geometry, we set the velocity of the fluid to be 0 everywhere (this
is the default value so we do not have to specify this explicitly), except for
the first row at the top, where we set the fluid to have ``max_v`` velocity
in the horizontal direction. It is important to always use an index expression
when assigning to ``sim.rho`` or ``sim.vx``, etc.::

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v

At this point, we are almost good to go. The only remaining thing to do is to
instantiate the :class:`LBSimulationController` class from the :mod:`controller`
providing :class:`LDCSim` as an argument.  Now we only have to run the simulation::

    ctrl = LBSimulationController(LDCSim)
    ctrl.run()

How it works behind the scenes
------------------------------
When the :func:`controller.LBSimulationController.run` method is called, Sailfish
instantiates a controller object, which is responsible for setting up and managing
the simulation.  All this normally happens "behind the scenes" so that you probably
do not need to worry about the details (check out the :ref:`internals` section
for the details).  The most important thing in this process is code generation.
Sailfish uses the Mako template engine and the information about your specific
simulation to generate optimized CUDA C or OpenCL code. The generated code is then compiled on the
fly by the :mod:`pyopencl` or :mod:`pycuda` modules into a binary which is executed on the GPU.

The template for the compute unit source is contained in the ``.mako`` files in the ``templates``
directory of the :mod:`sailfish` module.  It is written in a mix of Python, Mako and CUDA C.
Parts of the code that end up in GPU functions are also generated by the :mod:`sym` module.
This module contains mainly functions which return SymPy expressions, which are then converted to C code.
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
Most of the classes that take in some form part in a Sailfish simulation can define their own
command line parameters, which can be used to easily control the simulation. To get a full
list of currently supported options, run any Sailfish simulation with ``--help``.
Some of the basic settings you might want to play with when starting to work with Sailfish
are as follows:

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
* ``--every=N``: transfer data from the GPU to the host every ``N`` iterations.
* ``--output=FILE``: base name of the file to which the results of the simulation are to be
  saved.  The default format is npz (numpy).
* ``--max_iters=N``: the number of iterations the simulation is to be run for.

The ``--save_src`` option is particularly useful if you want to learn the basic structure of the
GPU code.  The Mako template files, which contain the actual code, can be difficult to
understand at first.  To avoid this complexity, you might want to save the generated compute device code
and inspect it in a text editor.  The generated code will be automatically formatted to be
readable unless the ``--noformat_src`` option is specified.  The command used to format the
code is hardcoded in the :mod:`codegen` module, requires the ``indent`` utility, and is set
so that the generated code roughly follows the formatting style of the Linux kernel
(with longer lines, which can be useful for complex expressions).

Troubleshooting
---------------

Debugging Sailfish programs in an interactive debugger.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, Sailfish simulations are run in multiple processes regardless of whether more than
one subdomain is used or not. This can present a challenge to programs such as pudb, which will
not be able to easily cross the subprocess boundary. The ``--debug_single_process`` option can
be used to force the controller, master and subdomain runner to run in a single process.
Note that only one subdomain is allowed in this mode, and that the visualization code
will still run in a separate process.

My simulation works fine in single precision, but breaks in double precision.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If your simulation runs in double precision, but generates clearly unphysical results that
do not appear when it's run in single precision, it's possible that the CUDA optimizing compiler
is generating broken code.  To check whether this is the case, you need to disable all optimizations
by running your simulation with the ``--cuda-nvcc-opts="-Xopencc -O0"`` command line option.
Note that this will significantly decrease the performance of your simulation.

