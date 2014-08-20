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
|                           | incompressible LBGK [JCP98]            |
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
.. [JPC98] He, Xiaoyi, Shiyi Chen, and Gary D. Doolen. "A novel thermal model for the lattice Boltzmann method in incompressible limit." Journal of Computational Physics 146.1 (1998): 282-300.

Boundary conditions
-------------------
Check the :mod:`node_type` module for a full and up-to-date list of supported boundary
conditions.
