<%namespace file="utils.mako" import="get_field_off"/>

${device_func} inline void laplacian_and_grad(${global_ptr} float *field, int i, float *laplacian, float *grad, int x, int y
%if dim == 3:
	, int z
%endif
)
{
	int off, nx, ny;
	%if dim == 3:
		int nz;
	%endif

	${get_field_off(+1, 0, 0)}	float fe = field[i + off];
	${get_field_off(-1, 0, 0)}	float fw = field[i + off];
	${get_field_off(0, +1, 0)}	float fn = field[i + off];
	${get_field_off(0, -1, 0)}	float fs = field[i + off];
	%if dim == 3:
		${get_field_off(0, 0, +1)}	float ft = field[i + off];
		${get_field_off(0, 0, -1)}	float fb = field[i + off];
		laplacian[0] = fe + fw + fn + fs + ft + fb - 6.0f * field[i];
		grad[2] = (ft - fb) / 2.0f;
	%elif dim == 2:
		laplacian[0] = fe + fw + fn + fs - 4.0f * field[i];
	%endif

	grad[0] = (fe - fw) / 2.0f;
	grad[1] = (fn - fs) / 2.0f;
}

