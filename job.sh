#!/bin/bash
#PBS -S /bin/bash
#PBS -N trainlog
# This wants full node (full IO, full CPU, and full GPU)
#PBS -l select=1:ncpus=1:mem=10Gb
#PBS -l place=excl
#PBS -l walltime=1:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR/"
echo  "START:"$(date)
echo "PWD: "$(pwd)
echo "HOST: "$(hostname)
echo "CONFIG: $CONFIG"
docker run -v $PWD:/tf -w /tf --gpus all -i --rm tensorflow/tensorflow@sha256:3f8f06cdfbc09c54568f191bbc54419b348ecc08dc5e031a53c22c6bba0a252e\
            bash kernel.sh $CONFIG
echo  "END:"$(date)

