Installation
============

Sailfish requires no installation and all sample simulations provided in the executable
.py files can simply be started from a shell, provided the required packages are
installed on the host system.  These are as follows (minimal required versions are shown):

General requirements:

* numpy-1.6.0
* sympy-0.6.5
* mako-0.2.5
* execnet-1.0.9
* Python zeromq-2.1.10
* netifaces-0.6 (for distributed simulations only)
* a Python module for the computational backend (one or more of the following):

  * pycuda-0.92 (with the NVIDIA drivers and NVIDIA CUDA Toolkit)
  * pypencl (with any OpenCL implementation)

Visualization (optional):

* pygame, scipy (for 2D/3D)
* mayavi (for 3D)

Data output (optional):

* tvtk (VTK output)

Tests:

* matplotlib

Versions older than those listed above might also work, but have not been tested.

Downloading Sailfish
--------------------

We currently do not provide snapshot tarballs of the code, so you will need to get Sailfish
directly from its git repository::

  git clone git://gitorious.org/sailfish/sailfish.git

Sailfish milestones and releases are appropriately tagged in the repository.  We try to
make sure the code is always in a working state, but if you find the most recent checkout
to be somehow broken, you might want to rewind to one of the tagged releases, e.g.::

  git checkout v0.1-alpha1

Gentoo installation instructions
--------------------------------

To install the required packages on a Gentoo system::

  emerge numpy scipy pytables mayavi matplotlib mako pygame pycuda sympy dev-util/git

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

  apt-get install gcc-4.4 g++-4.4 python-pygame mayavi2 python-matplotlib python-numpy python-tables python-scipy python-sympy
  apt-get install python-mako python-decorator python-pytools  build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev
  apt-get install git-core
  git clone http://git.tiker.net/trees/pycuda.git
  cd pycuda
  git submodule init
  git submodule update
  ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/local/cuda/lib64 --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-mt --boost-thread-libname=boost_thread-mt
  make -j4
  python setup.py install

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

The easiest way to install all the Sailfish prerequisites on Mac OS X is to use the Mac Ports
project and the PyOpenCL backend in Sailfish.  Follow the instructions at http://www.macports.org/,
and then run::

  port install py26-sympy py26-pyopencl py26-game py26-mako py26-scipy

To run the Sailfish examples, remember to use the correct Python interpreter (i.e. the one
installed via Mac Ports).  For instance:

  python2.6 ./lbm_ldc.py --scr_depth=24

Note that for the pygame visualization, you may need to specify the --scr_depth option.
