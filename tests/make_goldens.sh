#!/bin/bash
#
# Runs every example for a few iterations, saves the output, and compares
# it against a set of goldens.
#
# Usage:
#  make_goldens.sh <output_dir> <golden_dir>

output_dir=$1

[[ -z ${output_dir} ]] && exit 1

source tests/goldens.sh
collect_data ${output_dir}
