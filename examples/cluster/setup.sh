#!/bin/bash

# A sample script to install Python 2.7 and all Sailfish dependencies in $PREFIX.
#
# Adjust the 3 variables below to match your environment.  If you have
# administrative privileges for your system, prefer to install the required
# software from distribution packages.  This script is designed to make it easy
# to run Sailfish on clusters where all setup has to be done by the user.

export CUDA_ROOT=/usr/local/cuda
export PATH=${CUDA_ROOT}/bin:$PATH
export PREFIX=/storage/$USER

mkdir src
cd src

# Python
wget -c http://www.python.org/ftp/python/2.7.3/Python-2.7.3.tar.bz2
tar jxf Python-2.7.3.tar.bz2
cd Python-2.7.3
./configure --prefix=${PREFIX}
make
make install
cd ..

export PATH="${PREFIX}/bin:$PATH"
export CFLAGS="${CFLAGS} -Wl,-rpath -Wl,${PREFIX}/lib -I${PREFIX}/include -L${PREFIX}/lib"
export CXXFLAGS="${CFLAGS} -Wl,-rpath -Wl,${PREFIX}/lib -I${PREFIX}/include -L${PREFIX}/lib"

#wget -c http://archive.ubuntu.com/ubuntu/pool/main/u/util-linux/util-linux_2.20.1.orig.tar.gz
#tar zxf util-linux_2.20.1.orig.tar.gz
#cd util-linux-2.20.1/
#./configure --prefix=${PREFIX}
#make
#make install
#cd ..

# zmq
wget -c http://download.zeromq.org/zeromq-3.2.2.tar.gz
tar zxf zeromq-3.2.2.tar.gz
cd zeromq-3.2.2
./configure --prefix=${PREFIX}
make
make install
cd ..

wget -c http://pypi.python.org/packages/source/p/pyzmq/pyzmq-2.2.0.1.tar.gz
tar zxf pyzmq-2.2.0.1.tar.gz
cd pyzmq-2.2.0.1
python setup.py install --prefix=${PREFIX} --zmq=${PREFIX}
cd ..

wget -c http://pypi.python.org/packages/source/p/pip/pip-1.2.1.tar.gz
tar zxf pip-1.2.1.tar.gz
cd pip-1.2.1
python setup.py install --prefix=${PREFIX}
cd ..

wget -c http://pypi.python.org/packages/source/d/distribute/distribute-0.6.34.tar.gz
tar zxf distribute-0.6.34.tar.gz
cd distribute-0.6.34
python setup.py install --prefix=${PREFIX}
cd ..

export ATLAS=None

pip install distribute
pip install sympy
pip install mako
pip install execnet
pip install netifaces
pip install numpy
pip install scipy
pip install ipython
pip install pycuda
pip install blosc
