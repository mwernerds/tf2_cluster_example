# Cluster Mini-HowTo

## Running a job

```
qsub -q training job.sh
```

Giving it arguments always through env.


```
 qsub -q training -v CONFIG="configs/config.json"  job.sh
```


## Why JSON evaluation?

Because JSON is our ubiquitous tool to solve all problems, for example,
```
 ./results.sh  |jq -s ' max_by(.testprec)'
```
or if we want to change the code
```
qstat -f -F json |jq -r '.Jobs|keys[]'   | parallel qdel {}
```