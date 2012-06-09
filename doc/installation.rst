Installation
============

Sailfish requires no installation and all sample simulations provided in the executable
.py files can simply be started from a shell, provided the required packages are
installed on the host system.  These are as follows (minimal required versions are shown):

General requirements:

* numpy-1.6.0
* sympy-0.7.0
* mako-0.2.5
* execnet-1.0.9
* Python zeromq-2.1.10
* netifaces-0.6 (for distributed simulations only)
* a Python module for the computational backend (one or more of the following):

  * pycuda-0.92 (with the NVIDIA drivers and NVIDIA CUDA Toolkit)
  * pypencl (with any OpenCL implementation)

Visualization (optional):

* pygame
* matplotlib

Data output (optional):

* tvtk (VTK output)

Regression tests:

* matplotlib

Versions older than those listed above might also work, but have not been tested.

Downloading Sailfish
--------------------

We currently do not provide snapshot tarballs of the code, so you will need to get Sailfish
directly from its git repository::

  git clone git://github.com/sailfish-team/sailfish.git

Sailfish milestones and releases are appropriately tagged in the repository.  We try to
make sure the code is always in a working state, but if you find the most recent checkout
to be somehow broken, you might want to rewind to one of the tagged releases, e.g.::

  git checkout v0.2

Gentoo installation instructions
--------------------------------

To install the required packages on a Gentoo system::

  emerge numpy scipy matplotlib mako pygame pycuda sympy dev-util/git

You can also replace ``pycuda`` with ``pyopencl`` if you wish to use the OpenCL backend
in Sailfish.

CUDA 4.0 requires GCC 4.4 or older.  If you are using GCC 4.5+ as the main compiler on
your system, you can still run Sailfish simulations without changing any global settings
by adding::

  --cuda-nvcc-opts="-ccbin /usr/x86_64-pc-linux-gnu/gcc-bin/4.4.6/"

when calling the simulation (replace 4.4.6 with the actual version of the GCC installed
on your system and use ``i686`` instead of ``x86_64`` if you are on a 32-bit machine).

Ubuntu installation instructions
--------------------------------

These instructions assume that you want to use the CUDA backend.  Before installing following them,
please make sure the NVIDIA CUDA Toolkit is installed in ``/usr/local/cuda`` (default location) and
that ``nvcc`` is in your ``$PATH``.  You can get the necessary files at http://nvidia.com/cuda.

To install the required packages on an Ubuntu system::

  apt-get install python-pygame mayavi2 python-matplotlib python-numpy python-scipy python-sympy python-zmq python-execnet
  apt-get install python-mako python-decorator python-pytools build-essential python-dev python-setuptools libboost-dev libboost-python-dev libboost-thread-dev
  apt-get install python-pip git-core
  sudo PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH pip install pycuda

For 32-bit systems please change ``/usr/local/cuda/lib64`` to ``/usr/local/cuda/lib``.

There are currently no packages for PyCUDA/PyOpenCL available for
Ubuntu, so these have to be installed manually from a checked-out upstream code repository of
these projects or a snapshot tarball (as illustrated above for PyCUDA).  If this method does not
work for you, please refer to http://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu for further
instructions about installing PyCUDA on Ubuntu.

Please also note that the NumPy version provided in Ubuntu releases older than Karmic is not
recent enough for Sailfish.

When running Sailfish simulations, you can use::

  --cuda-nvcc-opts="--compiler-bindir=/usr/bin/gcc-4.4"

to avoid compatibility problems with the CUDA compiler, which as of CUDA 4.0 only supports GCC 4.4 and older.


Mac OS X installation instructions (Mac Ports)
----------------------------------------------

The easiest way to install all the Sailfish prerequisites on Mac OS X is to use the MacPorts
project and the PyOpenCL backend in Sailfish.  Follow the instructions at http://www.macports.org/,
and then run::

  port install py-sympy py-pyopencl py-game py-mako py-scipy py-zmq

To run the Sailfish examples, remember to use the correct Python interpreter (i.e. the one
installed via MacPorts).  For instance:

  python2.7 ./ldc_2d.py --mode=visualization --visualize=pygame --scr_depth=24

Note that for the pygame visualization, you may need to specify the --scr_depth option.
