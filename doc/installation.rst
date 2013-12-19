Installation
============

Sailfish requires no installation and all sample simulations provided in the executable
.py files can simply be started from a shell, provided the required packages are
installed on the host system.  These are as follows (minimal required versions are shown):

General requirements:

* numpy-1.7.0
* scipy-0.13.0
* sympy-0.7.3
* mako-0.9.0
* execnet-1.0.9
* pyzmq-14.0.0
* netifaces-0.8 (for distributed simulations only)
* a Python module for the computational backend (one or more of the following):

  * pycuda-2013.1 (with the NVIDIA drivers and NVIDIA CUDA Toolkit)
  * pypencl (with any OpenCL implementation)

Visualization (optional):

* pygame
* matplotlib
* wxpython

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

  git checkout 2012.1

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

On Ubuntu 13.10 or later all required dependencies can be installed from the standard
package repository::

  apt-get install nvidia-current-dev nvidia-current-updates nvidia-current-updates-dev nvidia-cuda-dev
  apt-get install python-pycuda python-numpy python-matplotlib python-scipy python-sympy
  apt-get install python-zmq python-execnet git

These instructions assume that you want to use the CUDA backend.  On older Ubuntu versions
some packages might require manual installation (we recommend the ``pip`` Python package
installer).

When running Sailfish simulations, you can use::

  --cuda-nvcc-opts="--compiler-bindir=/usr/bin/gcc-4.4"

to avoid compatibility problems with the CUDA compiler, which as of CUDA 4.0 only supports GCC 4.4 and older.

Installation instructions for other Linux systems
-------------------------------------------------

Whenever possible, it is recommended to install all necessary dependencies through the
native package manager for your distribution. If the package manager lacks a dependency
or if you do not have admin privileges allowing a global installation, we recommend the
``pip`` Python package installer, which can currently install most Sailfish dependencies.

Please refer to the ``examples/cluster/setup.sh`` script in our code repository
for sample invocations. Note that this script assumes a completely bare system without
even a Python interpreter.

Mac OS X installation instructions (Mac Ports)
----------------------------------------------

The easiest way to install all the Sailfish prerequisites on Mac OS X is to use the MacPorts
project and the PyOpenCL backend in Sailfish.  Follow the instructions at http://www.macports.org/,
and then run::

  port install py-sympy py-pyopencl py-game py-mako py-scipy py-zmq

To run the Sailfish examples, remember to use the correct Python interpreter (i.e. the one
installed via MacPorts).  For instance::

  python2.7 ./ldc_2d.py --mode=visualization

If this fails, try running the simulation without any arguments to see whether the problem
is with the visualization module or the computational backend.
