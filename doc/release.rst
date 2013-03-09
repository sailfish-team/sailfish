Release notes
=============

2013.1
""""""
 * 4 new outflow boundary conditions: Grad's approximation (NTGradFreeflow), Yu's method (NTYuOutflow), node copying method (NTCopy), Neumann's BC (NTNeumann).
 * New boundary conditions based on the regularized LB dynamics (NTRegularizedDensity, NTRegularizedVelocity).
 * New no-slip boundary condition based on the Tamm-Mott-Smith approximation (NTWallTMS).
 * NTHalfBBWall is now supported when the AA memory layout is used.
 * Added support for the exact difference method (EDM) of applying body forces.
 * Added support for a model minimizing round-off errors (--minimize_roundoff)
 * New examples: four rolls mill, 2D lid-driven cavity with the entropic model.
 * Restored the cutplane visualization engine for 3D simulations to a working state.
 * Added a mechanism of easy inclusion of additional code fragments (commit ae7bdff4)
 * Added support for regularized LB dynamics (--regularized).
 * Fixed the entropic LB model.
 * Compatiblity fixed for sympy-0.7.2 and numpy-1.7.0.
 * Added support for distributed simulations on LSF clusters.
 * Added the D3Q27 lattice.

2012.2
""""""
 * Added support for dynamic forces (time-dependent, or dependent on the macroscopic fields).

2012.1
""""""
 * Added support for the AA access pattern, which provides substantial memory savings (~50%).
 * Non-fluid nodes are masked in the output fields.
