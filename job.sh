#!/bin/bash
#PBS -S /bin/bash
# This wants full node (full IO, full CPU, and full GPU)
#PBS -l select=1:ncpus=1:mem=10Gb
#PBS -l place=excl
#PBS -N trainlog
#PBS -l walltime=00:01:00
#PBS -j oe

# walltime in seconds

WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
WALLTIME_SECONDS=$(echo "$WALLTIME" | sed -E 's/(.*):(.+):(.+)/\1*3600+\2*60+\3/;s/(.+):(.+)/\1*60+\2/' | bc)
SLEEPTIME=$(($WALLTIME_SECONDS - 10))

echo "Spawning for $SLEEPTIME/${WALLTIME_SECONDS} seconds"

cd "$PBS_O_WORKDIR/"
echo  "START:"$(date)
echo "PWD: "$(pwd)
echo "HOST: "$(hostname)
echo "CONFIG: $CONFIG"
CONTAINER_ID=$(docker  run -d --env HDF5_USE_FILE_LOCKING=FALSE --user=$(id -u):$(id -g) -v $PWD:/tf -w /tf --gpus all -i --rm mwernerds/tf_example@sha256:5d8e6b7b315b0859b4a69abe51a1ea5dd4214132a217f995d28029051e3705bd\
		       python3 train.py $CONFIG)
echo "Container ID: ${CONTAINER_ID}"
docker logs $CONTAINER_ID --follow &

for I in $(seq 0 10 $SLEEPTIME); do
    echo Inspecting after $I
    if ! docker inspect $CONTAINER_ID -f '{{.State.Running}}' > /dev/null 2> /dev/null; then
	echo "STATUS: Container died roughly after $I seconds. Exiting assuming succes"
	exit 0;
    fi
    sleep 10s;
done
echo "KILL AT: $(date)"
docker stop "$CONTAINER_ID"
echo  "END:"$(date)

