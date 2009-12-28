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

Gentoo installation instructions
--------------------------------

To install the required packages on a Gentoo system::

  emerge numpy scipy pytables mayavi matplotlib mako pygame pycuda sympy

Ubuntu installation instructions
--------------------------------

To install the required packages on an Ubuntu system::

  apt-get install python-pygame mayavi2 python-matplotlib python-numpy python-tables python-scipy python-mako

There are currently no (recent enough) packages for sympy and pycuda/pyopencl avaiable for
Ubuntu, so these have to be installed manually from a checked-out upstream code repository of
these projects or a snapshot tarball.


