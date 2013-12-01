#!/bin/bash

# ldc_2d_entropic: expected to generate differences in entropy estimate at
#                  boundary nodes where not all distributions are defined

blacklist="examples/ldc_2d_unorm.py examples/duct_flow.py examples/ldc_2d_entropic.py examples/binary_fluid/fe_poiseuille_2d.py examples/binary_fluid/fe_capillary_wave_2d.py" # examples/binary_fluid/sc_poiseuille_2d.py"
tmpdir=$(mktemp -d)

# Disable FMAD as it generates spurious differences for Shan-Chen models. This is casued
# by how registers are allocated when computing the density-weighted velocity in the
# PrepareMacroFields step.
common="--cuda-nvcc-opts=-fmad=false --every=50 --seed=1234 --quiet --max_iters=50"

find examples -perm +0111 -name '*.py' | while read filename ; do
	if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
		echo -n "Testing ${filename}..."
		python $filename --access_pattern=AB ${common} --output=${tmpdir}/result_ab
		python $filename --access_pattern=AA ${common} --output=${tmpdir}/result_aa

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
