#PBS -S /bin/bash
#PBS -q training
#PBS -N trainlog
#PBS -l nodes=1:ppn=1
#PBS -l mem=1gb
#PBS -l walltime=1:00:00
#PBS -j oe


#function die(){
#echo "FATAL: $@"
#exit -1
#}
#
#if test "xPBS_O_WORKDIR" != "x";then
#    # we are on queue system, here we must prepare nodes
#    if test "x$CONFIG" == "x"; then
#	echo "You need to run with CONFIG environment variable."
#	exit -2;
#    fi
#    
#else
#    CONFIG="$1"
#fi
#   
#test -f "$CONFIG" || die "Configuration file not found on job execution"
#


# With image pinning

# only set workdir if really on PBS
# execute
cd "$PBS_O_WORKDIR/"
echo  "START:"$(date)
echo "PWD: "$(pwd)
docker run -v $PWD:/tf -w /tf --gpus all -i --rm tensorflow/tensorflow@sha256:3f8f06cdfbc09c54568f191bbc54419b348ecc08dc5e031a53c22c6bba0a252e\
            bash kernel.sh $CONFIG
echo  "END:"$(date)

