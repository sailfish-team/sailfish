.. _internals:

Sailfish internals
==================

Architecture overview
---------------------

Simulation definition
~~~~~~~~~~~~~~~~~~~~~
Every Sailfish simulation is defined in terms of a simulation class,
typically inheriting from :class:`LBFluidSim` (or more generally, from
:class:`LBSim`).  In this class:

* additional simulation options can be defined in :func:`LBSim.add_options`
* default values for any simulation parameters can be specified in
  :func:`LBSim.update_defaults`
* simulation configuration can be altered after command-line parsing in
  :func:`LBSim.modify_config`
* additional setup (e.g. adding body forces) can be performed in the ``__init__``
  method (make sure to call the ``__init__`` method of all base classes at the
  beginning of your function)
* boundary and intitial conditions are defined by setting the :attr:`subdomain` attribute to a child of
  :class:`Subdomain` (typically :class:`Subdomain2D` and :class:`Subdomain3D`)

The global domain of a Sailfish simulation is always a rectangle (2D) or a
cuboid (3D), which automatically establishes a global Cartesian
coordinate system.  The distance between two neighboring lattice nodes
is the distance unit.  The domain can be sparse (not all nodes in the
cuboid exist) and subdivided into smaller units (subdomains).  This is accomplished
via a child class of :class:`LBGeometry` (or its specializations
:class:`LBGeometry2D`, :class:`LBGeometry3D`).  This class:

* specifies domain decomposition into subdomains (:func:`LBGeometry.subdomains`)
* defines options specific to the global simulation geometry
* provides access to the global domain size via the :attr:`LBGeometry2D.gx`,
  :attr:`LBGeometry2D.gy`, and :attr:`LBGeometry3D.gz` attributes

:func:`LBGeometry.subdomains` returns a list of :class:`SubdomainSpec` objects.
This makes it possible to build a refined, hierarchical grid (to be implemented;
by increasing node density in some subdomains) and to define a sparse domain
(by returning objects covering only a part of the global coordinate system).

Boundary conditions and initial values of macroscopic fields (density, velocity,
etc) are specified in the :func:`Subdomain.boundary_conditions` and
:func:`Subdomain.initial_conditions` methods, respectively.  These methods will
be called with ``hx``, ``hy`` and ``hz`` objects, which are numpy coordinate
arrays indicating nodes for which values are to be set.  The values in these
arrays are always in the *global* coordinate system.  The arrays should be used
as index objects when accessing field arrays in Sailfish or specifying
boundary conditions.  Your code in :func:`Subdomain.boundary_conditions` and
:func:`Subdomain.initial_conditions` should always define the global geometry and
make no assumptions about its division into subdomains.  In particular, the
Sailfish framework might arbitrarily subdivide your domain into multiple
subdomains to distribute the work among many computational units.

Simulation execution
~~~~~~~~~~~~~~~~~~~~
Sailfish is designed to run fluid simulations in a distributed and hybrid
environment, spreading work between multiple machines and GPUs.

The simulation execution begins with an instance of :class:`LBSimulationController`.

From the command line to a running simulation
---------------------------------------------

This section explains what happens in the first few seconds after you
start executing your simulation script and before the simulation is
actually running.

Distributed execution
---------------------

A distributed simulation is started by the controller mapping subdomains to
available nodes (as specified in a cluster definition file).  This is followed
by establishing an SSH connection to all nodes to which at least one block has been
assigned.  Once the connection is established, the ``execnet`` module is used to 
(optionally) sync files from the controller host to the node, and to execute the
:func:`_start_cluster_machine_master` function to start a :class:`LBMachineMaster`
on each node.  The masters and the controller are then linked by an execnet channel.

Each master starts a :class:`LBBlockRunner` for each of its subdomains.  The runners
are executed as subprocesses, and they communicate with the master using zeromq
IPC connections.  For each connected subdomain pair, one of the subdomains starts a listening
zeromq TCP socket with a random port.  This port is then communicated to the master,
which forwards it to the controller.  Once all runners have started, the controller
builds a global port map, which is then sent through the masters to all runners, which
use it to establish two-way connections between all connected subdomain pairs.

Inside a simulation
-------------------

This section explains the data structures and data flow of a live
simulation.

Template overview and conventions
---------------------------------

Specifying configuration options
--------------------------------

