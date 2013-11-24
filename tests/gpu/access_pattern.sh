#!/bin/bash

blacklist="examples/boolean_geometry.py examples/ldc_2d_unorm.py examples/duct_flow.py examples/ldc_2d_entropic.py examples/binary_fluid/fe_poiseuille_2d.py examples/binary_fluid/fe_capillary_wave_2d.py examples/binary_fluid/sc_poiseuille_2d.py"
tmpdir=$(mktemp -d)

find examples -perm +0111 -name '*.py' | while read filename ; do
	if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
		echo -n "Testing ${filename}..."
		python $filename --max_iters=50 --access_pattern=AB --every=50 --seed=1234 --quiet --output=${tmpdir}/result_ab
		python $filename --max_iters=50 --access_pattern=AA --every=50 --seed=1234 --quiet --output=${tmpdir}/result_aa

		if utils/compare_results.py ${tmpdir}/result_ab.0.50.npz ${tmpdir}/result_aa.0.50.npz ; then
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
