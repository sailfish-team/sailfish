#!/bin/bash

blacklist="examples/boolean_geometry.py"

find examples -perm +0111 -name '*.py' | while read filename ; do
	if [[ ${blacklist/${filename}/} == ${blacklist} ]]; then
		echo -n "Testing ${filename}..."
		if ! python $filename --max_iters=10 --quiet ; then
			echo "failed"
		else
			echo "ok"
		fi
	fi
done
