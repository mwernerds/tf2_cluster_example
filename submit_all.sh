#!/bin/bash


for f in $(ls configs/*.json | sort -R); do
    echo "Trying $f";
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
