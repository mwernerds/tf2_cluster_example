# Cluster Mini-HowTo

## Running a job

```
qsub -q training job.sh
```

Giving it arguments always through env.


```
 qsub -q training -v CONFIG="configs/config.json"  job.sh
```
