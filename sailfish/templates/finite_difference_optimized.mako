<%namespace file="utils.mako" import="get_field_off"/>

// More sophisticated finite difference formulas optimized to minimize spurious velocities
// at the surface of a spherical drop.  Fomulas taken from:
//   Phys Rev E 77, 046702 (2008)
${device_func} inline void laplacian_and_grad(${global_ptr} float *field, int fi, int i, float *laplacian, float *grad, int x, int y
%if dim == 3:
	, int z
%endif
)
{
	int off;

	%if block.envelope_size == 0:
		__UNIMPLEMENTED__
	%endif

	${get_field_off(+1, 0, 0)}	float fe = field[i + off];
	${get_field_off(-1, 0, 0)}	float fw = field[i + off];
	${get_field_off(0, +1, 0)}	float fn = field[i + off];
	${get_field_off(0, -1, 0)}	float fs = field[i + off];
	${get_field_off(+1, +1, 0)}	float fne = field[i + off];
	${get_field_off(-1, +1, 0)}	float fnw = field[i + off];
	${get_field_off(+1, -1, 0)}	float fse = field[i + off];
	${get_field_off(-1, -1, 0)}	float fsw = field[i + off];

%if dim == 3:
	${get_field_off(0, 0, +1)}	float ft = field[i + off];
	${get_field_off(0, 0, -1)}	float fb = field[i + off];
	${get_field_off(+1, 0, +1)}	float fte = field[i + off];
	${get_field_off(-1, 0, +1)}	float ftw = field[i + off];
	${get_field_off(+1, 0, -1)}	float fbe = field[i + off];
	${get_field_off(-1, 0, -1)}	float fbw = field[i + off];
	${get_field_off(0, +1, +1)}	float ftn = field[i + off];
	${get_field_off(0, -1, +1)}	float fts = field[i + off];
	${get_field_off(0, +1, -1)}	float fbn = field[i + off];
	${get_field_off(0, -1, -1)}	float fbs = field[i + off];

	grad[0] = (-fnw - fsw - ftw - fbw + fse + fne + fte + fbe) / 12.0f + (fe - fw) / 6.0f;
	grad[1] = (-fse - fsw - fts - fbs + fne + fnw + ftn + fbn) / 12.0f + (fn - fs) / 6.0f;
	grad[2] = (-fbe - fbw - fbn - fbs + fte + ftw + ftn + fts) / 12.0f + (ft - fb) / 6.0f;
	laplacian[0] = (fnw + fne + fse + fsw + fte + ftw + ftn + fts + fbe + fbw + fbn + fbs) / 6.0f  + (ft + fb + fe + fw + fn + fs) / 3.0f - 4.0f * field[i];
%else:
	grad[0] = (-fnw - fsw + fse + fne) / 12.0f + (fe - fw) / 3.0f;
	grad[1] = (-fse - fsw + fne + fnw) / 12.0f + (fn - fs) / 3.0f;
	laplacian[0] = (fnw + fne + fsw + fse + 4.0f * (fe + fw + fn + fs) - 20.0f * field[i]) / 6.0f;
%endif
}

