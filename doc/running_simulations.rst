Running simulations
===================

In this section, we show how to create a simple Lattice-Boltzmann simulation 
using Sailfish.
To keep things simple, we stick to two dimensions and use the lid-driven cavity
example, which is one of the standard test cases in computational fluid 
dynamics.

The program outline
-------------------
In order to build a Sailfish simulation, we create a new Python script. In this
script, we need to import the :mod:`lb_single`, :mod:`controller`, :mod:`geo_block`
and :mod:`geo` Sailfish modules::

    from sailfish import lb_single, controller, geo_block, geo

The :mod:`controller` module contains a class which will drive our simulation
described in the :class:`LDCSim` class based on :mod:`lb_single` module.
The :mod:`geo_block` module contains classes used to describe the geometry of the 
simulation and is used to define the boundary and initial conditions. The last
module :mod:`geo` determines the decompositon of the simulation area into subdomains.
    
Each single fluid Sailfish simulation is represented by a class derived
from :class:`lb_single.LBFluidSim` and :class:`lb_single.LBForcedSim`. 
In the simplest case, we just have to define a subdomain class::
	
    class LDCSim(LBFluidSim, LBForcedSim):
	subdomain = LDCBlock

:class:`LDCBlock` is a required class, derived from :class:`Subdomain2D`
or :class:`Subdomain3D` from the :mod:`geo_block` module, depending on the
dimensionality of the problem at hand. In our present case, we will use the 
former one. This class represents the geometry of the simulation. 

In :class:`LDCSim` class we can define the default parameter values and
additional command line arguments. To change the default parameters we have to
create the classmethod ``update_defaults`` where we update the ``defaults``
dictionary with the proper values of selected parameters. In our case we have to
change the size along the X axis (``lat_nx``) and the size along the Y axis
(``lat_ny``). This method is called before the class is instantiated.

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})

To add additional command line arguments we will create the classmethod
``add_options``. This method takes two arguments. ``group`` is a group of settings
connected with running the simulation, ``dim`` is the dimension of simulation
domain. This method, like ``update_defaults``, is called before the class is
instantiated. When the simulation is running, the command line arguments are
parsed and their settings are stored in ``self.config`` (using the standard
Python :py:mod:`argparse` module). In the first place, this method calls the same
methods in superclasses. After that we can add our options::

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)
        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')

Class :class:`LDCBlock` describes the simulation geometry and inherits from 
:class:`Subdomain2D`. The derived geometry class needs to define at least the
following two methods: ``bondary_conditions`` and ``initial_conditions``. 

``boundary_conditions`` is used to set the type of each node in the simulation
domain. The function takes two arguments: ``hx`` and ``hy``, which are NumPy
arrays constructed using the mgrid mechanism. We normally won’t be accessing these
parameters directly anyway, so the exact details of how the distributions are
stored are irrelevant at this point. The size of the simulation domain is already
known when the geometry class is instantiated and can be accessed via its
attributes ``gx`` (size along the X axis) and ``gy`` (size along the Y axis).

By default, the whole domain is initialized as fluid nodes. To define the
geometry, we need to redefine some of the nodes using the 
:const:`geo_block.Subdomain.NODE_WALL`, :const:`geo_block.Subdomain.NODE_VELOCITY`
or :const:`geo_block.Subdomain.NODE_PRESSURE` class constants. 
:const:`geo_block.Subdomain.NODE_WALL` represents a no-slip condition at a
stationary domain boundary. :const:`geo_block.Subdomain.NODE_VELOCITY` and 
:const:`geo_block.Subdomain.NODE_PRESSURE` represent a boundary condition with
specified velocity or pressure, respectively. To redefine the nodes, we will use
the ``set_node(location, type, data)`` function. Here, ``location`` is a NumPy
Boolean array. As for the remaining arguments of ``set_node``, ``type`` is one 
of the class constants discussed above, and data is an optional argument used to
specify the imposed velocity or pressure. 

In the lid-driven cavity (LDC) geometry, we consider a rectangular box, open at
the top where the fluid flows horizontally with some predefined velocity. We
therefore write our function as follows::

    class LDCBlock(Subdomain2D):
        max_v = 0.1

        def boundary_conditions(self, hx, hy):
            wall_map = np.logical_or(np.logical_or(hx == self.gx-1, hx == 0), hy == 0)
            self.set_node(hy == self.gy-1, self.NODE_VELOCITY, (self.max_v, 0.0))
            self.set_node(wall_map, self.NODE_WALL)

Now that we have the geometry out of the way, we can deal with the initial
conditions. This is done in the ``initial_conditions`` function, which is
responsible for setting the initial particle distributions in all nodes in the
simulation domain. The function takes three arguments: ``hx``, ``hy`` and 
``sim``. ``Sim`` is the reference to simulation object.

The way of specifying initial conditions is to provide the values of macroscopic
variables (density, velocity) everywhere in the simulation domain, and let the 
GPU calculate the equilibrium distributions.

In our LDC geometry, we set the velocity of the fluid everywhere to be 0 (this 
is the default value so we do not have to specify this explicitly), except for 
the first row at the top, where we set the fluid to have ``max_v`` velocity
in the horizontal direction. It is important to always use an index expression
when assigning to sim.rho or vx, etc. 

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v

At this point, we are almost good to go. The only remaining thing to do is to
instantiate the :class:`LBSimulationController` class from the :mod:'controller' 
module with two parameters: :class:`LDCSim` and :class:`LBGeometry2D` classes. The 
:class:`LBGeometry2D` class comes from the :mod:`geo` module. When we want to
create more specific decomposition of the domain into subdomains we can create a
class derived from that one. Now we only have to run the simulation::

    ctrl = LBSimulationController(LDCSim, LDCGeometry)
    ctrl.run()

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

