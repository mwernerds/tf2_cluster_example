for f in configs/*.json; do
    echo "Trying $f";
    #TODO test if model file is here, then skip and warn
    qsub  -v CONFIG="configs/config.json"  job.sh
done
qstat
