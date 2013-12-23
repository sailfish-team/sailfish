Release notes
=============

2013.1
""""""
 * 4 new outflow boundary conditions: Grad's approximation (NTGradFreeflow), Yu's method (NTYuOutflow), node copying method (NTCopy), Neumann's BC (NTNeumann).
 * New boundary conditions based on the regularized LB dynamics (NTRegularizedDensity, NTRegularizedVelocity).
 * New no-slip boundary condition based on the Tamm-Mott-Smith approximation (NTWallTMS).
 * NTHalfBBWall is now supported when the AA memory layout is used.
 * Exact difference method (EDM) of applying body forces.
 * Model minimizing round-off errors (--minimize_roundoff)
 * New examples: four rolls mill, 2D lid-driven cavity with the entropic model, external geometry from a npy file, Taylor-Green flow, duct flow, channel flow, capillary wave, two-phase Poiseuille flow.
 * Restored the cutplane visualization engine for 3D simulations to a working state.
 * Mechanism of easy inclusion of additional code fragments (commit ae7bdff4)
 * Regularized LB dynamics (--regularized).
 * Entropic LB model fixes.
 * Compatiblity fixes for sympy-0.7.2 and numpy-1.7.0.
 * Support for distributed simulations on LSF clusters.
 * D3Q27 lattice.
 * Support for drawing walls in the pygame visualization interface.
 * New on-line, remote visualization tool (utils/visualizer.py).
 * LinearlyInterpolatedTimeSeries data source for boundary conditions.
 * Indirect node addressing mode for sparse geometries.
 * Possibility to use neighbor link tagging instead of orientation (NTHalfBBWall in more complex geometries).
 * SIGHUP handling (creates a checkpoint dump).
 * Shuffle propagation (speeds up execution on some lower-end GPUs).
 * Force objects (on-GPU calculations of forces on solid objects).
 * On-GPU flow statistics (kinetic energy, Reynolds stresses, etc).

2012.2
""""""
 * Added support for dynamic forces (time-dependent, or dependent on the macroscopic fields).

2012.1
""""""
 * Added support for the AA access pattern, which provides substantial memory savings (~50%).
 * Non-fluid nodes are masked in the output fields.
