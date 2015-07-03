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

getpids() {
  pids=""
  tac ${logfile} | grep 'starting with PID' | while read line ; do
    [[ -n "$(echo ${line} | grep 'Machine master')" ]] && echo $pids && break
    pids="${pids} $(echo ${line} | sed -e 's/.* PID \([0-9]\+\).*/\1/')"
  done
}

pids=$(getpids)

if [[ -z "$(echo ${pids} | tr -c -d 0-9)" ]]; then
  echo "No SubdomainRunner PIDs found."
  exit 1
fi

echo "SubdomainRunner PIDs found: ${pids}"
sleep 10

while true; do
  remaining=$(qstat -f $PBS_JOBID | grep Walltime.Remaining | grep -o '[0-9]*')
  echo "Remaining seconds: ${remaining}"
  if [[ $remaining -lt $deadline ]]; then
    kill -SIGHUP ${pids}
    exit 0
  fi

  # Wait 5 minutes before checking again.
  sleep 300
done
