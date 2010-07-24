Motivation and Design Principles
================================

Sailfish is a general purpose fluid dynamics solver optimized for modern multi-core processors,
especially Graphics Processing Units (GPUs).  The solver is based on the Lattice Boltzmann Method,
which is conceptually quite simple to understand and which scales very well with increasing
computational resources.

The Sailfish project is also an experiment in scientific computing and software engineering.
Unlike the majority of CFD packages, which are written in compiled languages such as C++
or Fortran, Sailfish is implemented in Python and CUDA C/OpenCL.  We have found this
combination to be a very powerful one, making it possible to significantly shorten
development time without sacrificing any computational performance.

The general goals of the project are as follows:

* **Performance**: the code is optimized for the current generation of NVIDIA GPUs.
  With a single state-of-the-art video board, it is possible to achieve a simulation speed
  of 800 - 1200 MLUPS (depending on the used grid and model).  To achieve comparable performance with
  typical off-the-shelf CPUs, a small cluster would be necessary.

* **Scalability**: the code is designed to scale well with increasing number of compute cores.

* **Agility and extendability**: by implementing large parts of the code in a very
  expressive language (Python), we aim at encouraging rapid experimentation.
  Running tests, playing with new boundary conditions or new models is easy, and
  it often only requires changing a few lines of the kernel code.

* **Maintainability**: we keep the code clean and easy to understand.  The Mako
  template engine makes it possible to dynamically generate optimized code without
  any unnecessary cruft.

* **Ease of use**: defining new simulations and exploring simulation results is
  simple and many details are automated and by default hidden from the end-user.


