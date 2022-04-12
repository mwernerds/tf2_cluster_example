#!/bin/bash


for f in configs/*.json; do
    echo "Trying $f";
    #TODO test if model file is here, then skip and warn
    NAME=$(basename $f)
    if test -f "$NAME.h5"; then
	echo "Weight file <$NAME.h5> exists. Not running $f"
	continue;
    fi
    if test -f "models/$NAME.h5"; then
	echo "Weight file <$NAME.h5> exists. Not running $f"
	continue;
    fi
    
    qsub  -v CONFIG="configs/$NAME"  job.sh
done
qstat
