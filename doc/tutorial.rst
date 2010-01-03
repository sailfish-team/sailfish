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

    from sailfish import lbm, geo

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
two methods: ``define_nodes`` and ``init_dist``.

``define_nodes`` is used to set the type of each node in the simulation domain.  The
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
        def define_nodes(self):
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
TODO

Using the command-line arguments
--------------------------------
The base class for Sailfish simulations (``LBMSim``) defines a large number of command line
options which can be used to control the simulation.  To get a full list of currently supported
options, run any Sailfish simulation with the ``--help`` command line option.  Some of the
basic settings you might want to play with when starting to work with Sailfish are the following:

* ``--lat_w=N``, ``--lat_h=N``, ``--lat_d=N``: set lattice dimensions (width, height and depth, respectively)
* ``--precision=X``: set the precision of floating-point numbers used in the simulation (``single`` or ``double``).
  Note that double precision calculations will currently be significantly slower than their single precision
  counterparts and might not be supported at all on some older devices.
* ``--backend=X``: select the backend to be used to run the LBM simulation.  Supported values are
  ``cuda`` and ``opencl``.  Their availability will depend on the presence of required Python
  modules in the hsot system (pyopencl, pycuda).
* ``--save_src=FILE``: save the generated kernel code to ``FILE``
* ``--use_src=FILE``: use the kernel code from ``FILE`` instead of the one generated by Sailfish
  (useful for testing minor changes in the kernel code)
* ``--every=N``: update the display every ``N`` iterations
* ``--benchmark``: run the simulation in benchmark mode, printing information about its
  performance on the stadard output.
* ``--batch``, ``--nobatch``: force or disable batch mode, respectively.  In batch mode, all
  visualization modules are disabled and hooks defined for the simulation are run at
  specified iterations.  Batch mode requires specifying the ``max_iters`` option.
* ``--max_iters=N``: the number of iterations the simulation is to be run for in batch mode.

The ``--save_src`` option is particularly useful if you want to learn the basic structure of the
kernel code.  The ``lbm.mako`` file, which contains the actual code, can be difficult to
understand at first, as it mixes three languages: Python, the Mako template language and
CUDA C.  To avoid its complexity, you might want to save the generated compute device code
and inspect it in a text editor.  The generated code will generally be badly formatted,
as ``lbm.mako`` is optimized for its own readability, not the readability of the generated
code.  To make the generated code more readable, we suggest using the following commands::

    indent -linux -sob -l120 FILE
    sed -i -e '/^$/{N; s/\n\([\t ]*}\)$/\1/}' -e '/{$/{N; s/{\n$/{/}' FILE

This will reformat the generated code so that it roughly follows the formatting style
of the Linux kernel (with longer lines, which can be useful for complex expressions).
The ``sed`` call removes spurious empty lines.

