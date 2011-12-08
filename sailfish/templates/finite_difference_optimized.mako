<%namespace file="utils.mako" import="get_field_off,nonlocal_fld,fld_args"/>

${device_func} inline float get_field(${global_ptr} float *fx, int fi, int idx, int nx, int ny
%if dim == 3:
	, int nz
%endif
)
{
	return ${nonlocal_fld('x')};
}

// More sophisticated finite difference formulas optimized to minimize spurious velocities
// at the surface of a spherical drop.  Fomulas taken from:
//   Phys Rev E 77, 046702 (2008)
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
	${get_field_off(+1, +1, 0)}	float fne = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(-1, +1, 0)}	float fnw = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(+1, -1, 0)}	float fse = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(-1, -1, 0)}	float fsw = get_field(field, fi, i + off, ${fld_args()});

%if dim == 3:
	${get_field_off(0, 0, +1)}	float ft = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, 0, -1)}	float fb = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(+1, 0, +1)}	float fte = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(-1, 0, +1)}	float ftw = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(+1, 0, -1)}	float fbe = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(-1, 0, -1)}	float fbw = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, +1, +1)}	float ftn = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, -1, +1)}	float fts = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, +1, -1)}	float fbn = get_field(field, fi, i + off, ${fld_args()});
	${get_field_off(0, -1, -1)}	float fbs = get_field(field, fi, i + off, ${fld_args()});

	grad[0] = (-fnw - fsw - ftw - fbw + fse + fne + fte + fbe) / 12.0f + (fe - fw) / 6.0f;
	grad[1] = (-fse - fsw - fts - fbs + fne + fnw + ftn + fbn) / 12.0f + (fn - fs) / 6.0f;
	grad[2] = (-fbe - fbw - fbn - fbs + fte + ftw + ftn + fts) / 12.0f + (ft - fb) / 6.0f;
	laplacian[0] = (fnw + fne + fse + fsw + fte + ftw + ftn + fts + fbe + fbw + fbn + fbs) / 6.0f  + (ft + fb + fe + fw + fn + fs) / 3.0f - 4.0f * get_field(field, fi, i, x, y, z);
%else:
	grad[0] = (-fnw - fsw + fse + fne) / 12.0f + (fe - fw) / 3.0f;
	grad[1] = (-fse - fsw + fne + fnw) / 12.0f + (fn - fs) / 3.0f;
	laplacian[0] = (fnw + fne + fsw + fse + 4.0f * (fe + fw + fn + fs) - 20.0f * get_field(field, fi, i, x, y)) / 6.0f;
%endif
}

