#!/bin/bash
set -e
set -x
function die()
{
    echo "FATAL: $@"
    exit    
}

# Perform dataset splits
for dir in wf_*; do
    pushd $dir;
    test -d "train" && die "Train exists in $dir. Hence skipping"
    NFIRE=$(find fire | wc -l)
    NNOFIRE=$(find nofire| wc -l);
    echo "In $dir found $NFIRE fires and $NNOFIRE nofires"
    mkdir train
    mv fire train
    mv nofire train
    mkdir test test/fire test/nofire val val/fire val/nofire
    find  train/fire |grep -E 'png$'   | sort -R | head -1000 | parallel mv {} val/fire;
    find  train/nofire |grep -E 'png$'| sort -R | head -1000 | parallel mv {} val/nofire;

    find   train/fire  |grep -E 'png$' | sort -R | head -500 | parallel mv {} test/fire;
    find   train/nofire |grep -E 'png$'| sort -R | head -500 | parallel mv {} test/nofire;
   
    
    popd;
done
