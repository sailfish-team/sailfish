<%namespace file="kernel_common.mako" import="*"/>

${kernel} void ComputeEntropy(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} float *dist,
	${global_ptr} float *entropy
	${iteration_number_if_required()}
)
{
	${local_indices()}
	${load_node_type()}

	Dist d0;
	getDist(&d0, dist, gi ${iteration_number_arg_if_required()});

	entropy[gi] = CalculateEntropy(&d0);
}
