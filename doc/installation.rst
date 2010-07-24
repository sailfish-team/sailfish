Installation
============

Sailfish requires no installation and all sample simulations provided in the executable
.py files can simply be started from a shell, provided the required packages are
installed on the host system.  These are as follows (minimal required versions are shown):

General requirements:

* numpy-1.3.0
* sympy-0.6.5
* mako-0.2.5
* a Python module for the computational backend (one or more of the following):

  * pycuda-0.92 (with the NVIDIA drivers and NVIDIA CUDA Toolkit)
  * pypencl (with any OpenCL implementation)

Visualization (optional):

* pygame (for 2D/3D)
* mayavi (for 3D)

Data output (optional):

* pytables-2.1.1 (HDF5 output)
* tvtk (VTK output)

Tests:

* matplotlib

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

Ubuntu installation instructions
--------------------------------

To install the required packages on an Ubuntu system::

  apt-get install python-pygame mayavi2 python-matplotlib python-numpy python-tables python-scipy python-mako python-decorator
  apt-get install git-core python-setuptools libboost-dev
  git clone git://git.sympy.org/sympy.git
  git clone http://git.tiker.net/trees/pytools.git
  git clone http://git.tiker.net/trees/pycuda.git
  cd pytools
  python setup.py build
  python setup.py install
  cd ../sympy
  python setup.py build
  python setup.py install
  cd ../pycuda
  /configure.py --boost-python-libname=boost_python-mt-py26 --boost-thread-libname=boost_thread-mt --cuda-root=/usr/local/cuda
  python setup.py build
  python setup.py install

There are currently no (recent enough) packages for SymPy and PyCUDA/PyOpenCL available for
Ubuntu, so these have to be installed manually from a checked-out upstream code repository of
these projects or a snapshot tarball.

Please also note that the NumPy version provided in Ubuntu releases older than Karmic is not
recent enough for Sailfish.

The example listed above assumes that you want to use the CUDA backend.  Before installing pycuda,
please make sure the NVIDIA CUDA Toolkit is installed in ``/usr/local/cuda``.  You can get the
necessary files at http://nvidia.com/cuda.
