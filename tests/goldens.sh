#!/bin/bash

blacklist="examples/boolean_geometry.py examples/ldc_2d_unorm.py"

function collect_data() {
	local target_dir=$1

	find examples -perm +0111 -name '*.py' | while read filename ; do
		if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
			echo -n "Testing ${filename}..."
			if ! python $filename --max_iters=20 --every=20 --seed 1234 --quiet --output ${target_dir}/$(basename ${filename//.py/}); then
				echo "failed"
			else
				echo "ok"
			fi
		fi
	done

	rm ${target_dir}/*subdomains ${target_dir}/*00.npz
}

