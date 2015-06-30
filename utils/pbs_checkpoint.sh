#/bin/bash
#
# Usage:
#  ./pbs_checkpoint.sh <logfile> <seconds_remaining>
#
# Sends SIGHUP to the runner processes of a simulation in order to dump a
# checkpoint when less than <seconds_remaining> seconds are left in a PBS
# job.

if [[ -z "$PBS_JOBID" ]]; then
  echo "PBS job ID unknown."
  exit 1
fi

logfile=$1
deadline=$2

if [[ "$(cat ${logfile} | grep 'SubdomainRunner starting with PID' | wc -l)" -lt 1 ]]; then
  echo "No SubdomainRunner PIDs found."
  exit 1
fi

while true; do
  if [[ $(qstat -f $PBS_JOBID | grep Walltime.Remaining | grep -o '[0-9]*') -lt $deadline ]]; then
    cat ${logfile} | grep 'SubdomainRunner starting with PID ' | sed -e 's/.* PID \([0-9]\+\).*/\1/' | xargs kill -SIGHUP
    exit 0
  fi

  # Wait 5 minutes before checking again.
  sleep 300
done
