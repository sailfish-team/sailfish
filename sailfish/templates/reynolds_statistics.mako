## Kernels to compute various statistics of the hydrodynamic variables under
## Reynolds decomposition: f = f_mean + f'
##
## Profiles of (averaged in space):
##  <f>, <f^2>, <f^3>, <f^4>
##
## These can be used to compute:
##  variance sigma^2 = <(f')^2>
##  skewness <(f')^3> / sigma^3
##  flatness <(f')^4> / sigma^4
##
## Correlations:
##  <rho(y) u_i(y)>
##  <u_i u_j>
##
## Velocity component correlations can be used to compute the Reynolds stress components:
##  <u_i u_j> = <(u_i* + u_i') (u_j* + u_j')> =
##   = u_i* u_j* +			# known from single variable profiles: <u_i> * <u_j>
##	   u_i* <u_j'> +		# = 0 since <u_j'> = 0
##	   u_j* <u_i'> +		# = 0 since <u_i'> = 0
##	   <u_i' u_j'>			# Reynolds stress components
##
## where the star indicates the mean values.

<%namespace file="data_processing.mako" import="*"/>

// Computes <f>, <f^2>, <f^3>, <f^4>
${reduction('ComputeMoments', 0, num_inputs=1, stats=[[(0, 1)], [(0, 2)], [(0, 3)], [(0, 4)], block_size=512, out_type='double')}

// Inputs:
//  u_x, u_y, u_z, rho
//
// Computes
//  <u_x u_y>, <u_x u_z>, <u_y u_z>
//  <u_x rho>, <u_y rho>, <u_z rho>
${reduction('ComputeCorrelations', 0, num_inputs=4, stats=[
	[(0, 1), (1, 1)], [(0, 1), (2, 1)], [(1, 1), (2, 1)],
    [(0, 1), (3, 1)], [(1, 1), (3, 1)], [(2, 1), (3, 1)]
], block_size=256, out_type='double')}
