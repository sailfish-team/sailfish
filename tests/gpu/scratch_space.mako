<%def name="dummy()">
	int a
</%def>

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
${kernel_common.body(dummy)}
<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="mako_utils.mako" import="*"/>

// Writes data to the node scratch space.
${kernel} void TestNodeScratchSpaceWrite(
	${global_ptr} int *map
	${scratch_space_if_required()}
	)
{
	${local_indices_split()}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	if (!isNTGradFreeflow(type)) {
		return;
	}

	int scratch_id = decodeNodeScratchId(ncode);

	float buf[3];
	buf[0] = gx / 10.0f;
	buf[1] = gy / 10.0f;
	buf[2] = gx * gy / 10.0f;
	storeNodeScratchSpace(scratch_id, type, buf, node_scratch_space);
}

// Reads data from the node scratch space and saves it in output*.
${kernel} void TestNodeScratchSpaceRead(
	${global_ptr} int *map,
	${global_ptr} float *output_x,
	${global_ptr} float *output_y,
	${global_ptr} float *output_xy
	${scratch_space_if_required()}
	)
{
	${local_indices_split()}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	if (!isNTGradFreeflow(type)) {
		return;
	}

	int scratch_id = decodeNodeScratchId(ncode);

	float buf[3];
	loadNodeScratchSpace(scratch_id, type, node_scratch_space, buf);

	output_x[gi] = buf[0];
	output_y[gi] = buf[1];
	output_xy[gi] = buf[2];
}
