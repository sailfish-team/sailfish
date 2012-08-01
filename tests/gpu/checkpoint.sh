#!/bin/bash

sim=$1
tmpdir=$(mktemp -d)

echo -n "$sim .."

${sim} --max_iters=100 --final_checkpoint --checkpoint_file=${tmpdir}/tmp --quiet
${sim} --max_iters=200 --restore_from=${tmpdir}/tmp.0.cpoint.npz --output=${tmpdir}/result_2step --every=100 --quiet
${sim} --max_iters=200 --output=${tmpdir}/result_1step --every=200 --quiet

if utils/compare_results.py ${tmpdir}/result_1step.0.200.npz ${tmpdir}/result_2step.0.200.npz; then
	echo "ok"
else
	echo "failed"
	echo "Data in ${tmpdir}"
	exit 1
fi

rm ${tmpdir}/tmp* ${tmpdir}/result*
rmdir ${tmpdir}
