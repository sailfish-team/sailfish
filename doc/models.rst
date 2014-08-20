Supported features and models
=============================

+---------------------------+----------------------------------------+
| Feature type              | Supported variants                     |
+===========================+========================================+
| lattices                  | D2Q9                                   |
|                           +----------------------------------------+
|                           | D3Q13, D3Q15, D3Q19, D3Q27             |
+---------------------------+----------------------------------------+
| body forces               | Guo's method                           |
|                           +----------------------------------------+
|                           | Exact Difference Method                |
+---------------------------+----------------------------------------+
| relaxation dynamics       | LBGK                                   |
|                           +----------------------------------------+
|                           | MRT                                    |
|                           +----------------------------------------+
|                           | ELBM (entropic)                        |
|                           +----------------------------------------+
|                           | regularized                            |
+---------------------------+----------------------------------------+
| multicomponent models     | Shan-Chen                              |
|                           +----------------------------------------+
|                           | free energy [PRE78]_                   |
+---------------------------+----------------------------------------+
| turbulence                | Smagorinsky LES                        |
|                           +----------------------------------------+
|                           | ELBM                                   |
+---------------------------+----------------------------------------+
| other models              | single-phase Shan-Chen                 |
|                           +----------------------------------------+
|                           | shallow water                          |
|                           +----------------------------------------+
|                           | incompressible LBGK [JCP97]            |
+---------------------------+----------------------------------------+
| other features            | round-off minimization model           |
|                           +----------------------------------------+
|                           | checkpointing                          |
+---------------------------+----------------------------------------+
| distributed simulations   | ad-hoc (SSH)                           |
|                           +----------------------------------------+
|                           | PBS                                    |
|                           +----------------------------------------+
|                           | LSF                                    |
+---------------------------+----------------------------------------+
| precision                 | single                                 |
|                           +----------------------------------------+
|                           | double                                 |
+---------------------------+----------------------------------------+
| output formats            | numpy                                  |
|                           +----------------------------------------+
|                           | Matlab                                 |
|                           +----------------------------------------+
|                           | VTK                                    |
+---------------------------+----------------------------------------+
| computational backends    | CUDA                                   |
|                           +----------------------------------------+
|                           | OpenCL                                 |
+---------------------------+----------------------------------------+
| on-GPU statistics         | 1D profiles of the first 4 moments of  |
|                           | velocity and density                   |
|                           +----------------------------------------+
|                           | 1D profiles of correlations of         |
|                           | velocity components and density        |
|                           +----------------------------------------+
|                           | total kinetic energy and enstrophy     |
+---------------------------+----------------------------------------+

.. [PRE78] Contact line dynamics in binary lattice Boltzmann simulations, Phys. Rev. E 78, 056709 (2008). DOI: 10.1103/PhysRevE.78.056709
.. [JCP97] He, Xiaoyi, and Li-Shi Luo. "Lattice Boltzmann model for the incompressible Navier–Stokes equation." Journal of Statistical Physics 88.3-4 (1997): 927-944.

Boundary conditions
-------------------
Check the :mod:`node_type` module for a full and up-to-date list of supported boundary
conditions.
