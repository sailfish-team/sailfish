#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

classifiers = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Education',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
'Operating System :: MacOS :: MacOS X',
'Operating System :: POSIX :: Linux',
'Programming Language :: Python :: 2',
'Programming Language :: Python :: 2.7',
'Topic :: Scientific/Engineering :: Physics'
]


def do_setup():
    setup(name = 'sailfish',
          version = '2013.1',
          packages = ['sailfish'],
          install_requires = [
              'numpy >= 1.7.0',
              'scipy >= 0.13.0',
              'pycuda >= 2013.1.1',
              'pyopencl >= 2013.2',
              'netifaces >= 0.8',
              'pyzmq >= 14.0.0',
              'Mako >= 0.9.0',
              'sympy >= 0.7.3',
              'blosc >= 1.1',
          ],
          extras_require = {
              'doc': ['Sphinx >= 1.1', 'matplotlib >= 1.3.1'],
              'visualization': ['Pygame >= 1.7.1', 'wxPython >= 2.9.1.1'],
              'cluster': ['execnet >= 1.1'],
          },
          setup_requires = [ "setuptools_git >= 0.3", ],
          include_package_data = True,

          # Metadata.
          keywords = 'cuda opencl lbm fluid cfd lattice boltzmann computational',
          author = 'Michal Januszewski',
          author_email = 'sailfish-cfd@googlegroups.com',
          description = 'Multi-GPU implementation of the lattice Boltzmann method for CUDA/OpenCL.',
          classifiers = classifiers,
          license = 'LGPLv3',
          url = 'http://sailfish.us.edu.pl/',
         )

if __name__ == "__main__":
    do_setup()
