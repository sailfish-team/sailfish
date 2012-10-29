#!/bin/bash
#
# Runs every example for a few iterations, saves the output, and compares
# it against a set of goldens.
#
# Usage:
#  test_goldens.sh <output_dir> <golden_dir>

output_dir=$1
golden_dir=$2
blacklist="examples/boolean_geometry.py examples/ldc_2d_unorm.py"

[[ -z ${output_dir} || -z ${golden_dir} ]] && exit 1

find examples -perm +0111 -name '*.py' | while read filename ; do
	if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
		echo -n "Testing ${filename}..."
		if ! python $filename --max_iters=20 --every=20 --seed 1234 --quiet --output ${output_dir}/$(basename ${filename//.py/}); then
			echo "failed"
		else
			echo "ok"
		fi
	fi
done

rm ${output_dir}/*subdomains ${output_dir}/*00.npz

echo "Comparing $output_dir with goldens in $golden_dir"
for i in ${output_dir}/*.npz ; do
   	echo -n "$i..."
	if utils/compare_results.py $i ${i//$output_dir/$golden_dir} ; then
		echo "ok"
	fi
done
