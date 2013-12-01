#!/bin/bash

# ldc_2d_entropic: expected to generate differences in entropy estimate at
#                  boundary nodes where not all distributions are defined

blacklist="examples/ldc_2d_unorm.py examples/duct_flow.py examples/ldc_2d_entropic.py examples/binary_fluid/fe_poiseuille_2d.py examples/binary_fluid/fe_capillary_wave_2d.py examples/binary_fluid/sc_poiseuille_2d.py"
tmpdir=$(mktemp -d -t sailfish_addr.XXXXXXXX)

# Examples to run can be specified in a list on the command line as well.
targets="$*"
if [ -z "$targets" ]; then
	targets=$(find examples -perm +0111 -name '*.py')
fi

echo ${targets} | tr ' ' '\n' | while read filename ; do
	if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
		echo -n "Testing ${filename}..."
		python $filename --max_iters=50 --access_pattern=AB --node_addressing=direct --every=50 --seed=1234 --quiet --output=${tmpdir}/result_dir_ab
		python $filename --max_iters=50 --access_pattern=AB --node_addressing=indirect --every=50 --seed=1234 --silent --output=${tmpdir}/result_ind_ab
		python $filename --max_iters=50 --access_pattern=AA --node_addressing=indirect --every=50 --seed=1234 --silent --output=${tmpdir}/result_ind_aa

		if utils/compare_results.py ${tmpdir}/result_dir_ab.0.50.npz ${tmpdir}/result_ind_ab.0.50.npz && \
			utils/compare_results.py ${tmpdir}/result_dir_ab.0.50.npz ${tmpdir}/result_ind_aa.0.50.npz ; then
			echo "ok"
		else
			echo "failed"
			echo "Data in ${tmpdir}"
			exit 1
		fi
	fi
done || exit 1

rm ${tmpdir}/result*
rmdir ${tmpdir}
