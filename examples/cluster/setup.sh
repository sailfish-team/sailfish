#!/bin/bash

export STORAGE=/storage/$USER

wget http://download.zeromq.org/zeromq-2.2.0.tar.gz
tar zxf zeromq-2.2.0.tar.gz
cd zeromq-2.2.0
./configure --prefix=${STORAGE}
make
make install
cd ..
wget http://pypi.python.org/packages/source/p/pyzmq/pyzmq-2.2.0.tar.gz#md5=100b73973d6fb235b8da6adea403566e
tar zxf pyzmq-2.2.0.tar.gz
cd pyzmq-2.2.0
python setup.py install --prefix=${STORAGE} --zmq=${STORAGE}
cd ..

