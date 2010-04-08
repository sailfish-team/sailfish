<%namespace file="utils.mako" import="get_field_off,nonlocal_fld,fld_args"/>

${device_func} inline float get_field(${global_ptr} float *fx, int fi, int idx, int nx, int ny
%if dim == 3:
	, int nz
%endif
)
{
	if (0) { ; }
	%for fld_id in image_fields:
		else if (fi == ${fld_id}) {
			return ${nonlocal_fld(fld_id)};
		}
	%endfor
	else {
		return ${nonlocal_fld('x')};
	}

	return 0.0f;
}

${device_func} inline void laplacian_and_grad(${global_ptr} float *field, int fi, int i, float *laplacian, float *grad, int x, int y
%if dim == 3:
	, int z
%endif
)
{
	int off, nx, ny;
	%if dim == 3:
		int nz;
	%endif

	${get_field_off(+1, 0, 0)}	float fe = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(-1, 0, 0)}	float fw = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, +1, 0)}	float fn = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, -1, 0)}	float fs = get_field(field, fi, i + off, ${fld_args()});
	%if dim == 3:
		${get_field_off(0, 0, +1)}	float ft = get_field(field, fi, i + off, ${fld_args()});
		${get_field_off(0, 0, -1)}	float fb = get_field(field, fi, i + off, ${fld_args()});
		laplacian[0] = fe + fw + fn + fs + ft + fb - 6.0f * get_field(field, fi, i, x, y, z);
		grad[2] = (ft - fb) / 2.0f;
	%elif dim == 2:
		laplacian[0] = fe + fw + fn + fs - 4.0f * get_field(field, fi, i, x, y);
	%endif

	grad[0] = (fe - fw) / 2.0f;
	grad[1] = (fn - fs) / 2.0f;
}

