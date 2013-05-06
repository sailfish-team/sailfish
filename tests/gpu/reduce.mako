<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
${kernel_common.body(dummy)}
<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="data_processing.mako" import="*"/>

${reduction('TestX', 0, block_size=1024)}
${reduction('TestY', 1, block_size=1024)}

%if dim == 3:
	${reduction('TestZ', 2, block_size=1024)}
%endif

${reduction('TestY2', 1, block_size=32)}
