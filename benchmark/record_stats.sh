#!/bin/bash

# Records nvidia-smi stats (power draw, GPU utilization, memory bandwidth
# utilization, GPU memory use) while a Sailfish simulation is running.
#
# Usage:
# ./record_stats.sh <summary_file> <program_to_run>  

DEVICE=0
TMP=$(mktemp -d)

summary=$1
shift

$* &
PID=$!

while ps -p $! >&- ; do
	nvidia-smi -q -d POWER,UTILIZATION,MEMORY -i $DEVICE > $TMP/rec$(date +%s)
	sleep 2
done

cat $TMP/rec* | grep Draw | awk '{ print $4 }' > $TMP/ser_power
cat $TMP/rec* | grep Gpu | awk '{ print $3 }' > $TMP/ser_gpu_util
cat $TMP/rec* | grep ' Memory     ' | awk '{ print $3 }' > $TMP/ser_mem_util
cat $TMP/rec* | grep ' Used ' | awk '{ print $3 }' > $TMP/ser_mem_used

paste $TMP/ser_power $TMP/ser_gpu_util $TMP/ser_mem_util $TMP/ser_mem_used > ${summary}
rm -rf ${TMP}
