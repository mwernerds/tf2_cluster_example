#!/bin/bash
#PBS -S /bin/bash
#PBS -N trainlog
#PBS -l walltime=1:00:00
#PBS -j oe



cd "$PBS_O_WORKDIR/"
echo  "START:"$(date)
echo "PWD: "$(pwd)
echo "HOST: "$(hostname)
echo "CONFIG: $CONFIG"
docker  run --env HDF5_USE_FILE_LOCKING=FALSE --user=$(id -u):$(id -g) -v $PWD:/tf -w /tf --gpus all -i --rm mwernerds/tf_example@sha256:5d8e6b7b315b0859b4a69abe51a1ea5dd4214132a217f995d28029051e3705bd\
            python3 train.py $CONFIG
echo  "END:"$(date)

