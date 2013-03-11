#!/bin/bash
#
# Runs every example for a few iterations, saves the output, and compares
# it against a set of goldens.
#
# Usage:
#  test_goldens.sh <output_dir> <golden_dir>

output_dir=$1
golden_dir=$2

[[ -z ${output_dir} || -z ${golden_dir} ]] && exit 1

source tests/goldens.sh
collect_data ${output_dir}

echo "Comparing $output_dir with goldens in $golden_dir"
for i in ${output_dir}/*.npz ; do
   	echo -n "$i..."
	if utils/compare_results.py $i ${i//$output_dir/$golden_dir} ; then
		echo "ok"
	fi
done
