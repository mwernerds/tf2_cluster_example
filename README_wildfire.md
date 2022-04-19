# OpenEducationalResource Wildfire Detection

Our wildfire detection example is an example of how to employ big data management strategies in a simple setup
using consumer hardware in a reliable and sensible way. The aim of this approach is to get a clear and rather
complete picture of a given problem and to analyze it in an efficient, distirbuted, but simple way.

This document is more about the tooling and setup of the processing system and not a guide on how to
create the highest numbers and scores for your models.

## Wildfire Detection - A Typology of models
Wilffire detection is a complex task and can be formulated as a data mining problem in at least three common
settings:

- Pixel-Wise: For each pixel, decide on its own whether it is a fire candidate TODO:ADD CITATIONS
- Pixel-Context-Wise: For each pixel and summary statistics of a fixed surrounding, e.g., a window of NxN pixels, decide if it is fire
- Patch-Wise: For a patch of NxN pixels, decide whether there is fire inside
- Semantic Segmenation: For an input image of mxn pixels, create an output image of m x n scores, where high scores
  indicate fire.


## Patch-Wise Fire Detection

We start with a dataset of Sentinel 2 images for which three binary masks have been defined from existing
geodata in a preprocessing step: one mask for cloud-covered pixels, one for nodata pixels, and one for fire
pixels. This document is not about how to come up with this type of data, we are going to publish some soon.

More concretely, the dataset is given in many files, each possibly having a different size and different projection
following the Sentinel 2 grid which foresees a different UTM projection for any of the TODO different tiles in which
the Level 2 data (that is preprocessed and georeferenced) is provided to users.

The three masks (fire, clouds, nodata) as a layer into these images and are, therefore, having the same projection.

Now, we create fire patches which contain (1) no clouds (2) no nodata pixels and (3) no pixels that have been used before.
In order to organize this efficiently, we create a new mask as eight-bit integer in which
- 0 means good data of class nofire
- 1 means good data of class fire
- 2 means used in a patch
- 3 means cloud
- 4 means nodata

Then, we localize all pixels of fire using something like `np.where (mask == 1)`, extract a patch from the mask and continue
to extract data patches for the fire class if the whole patch smaller or equal to 1. Afterwards, we mask the whole patch area with a value of
two.
> Note that it might be a good idea to mask a larger area to further reduce the risk of spatial autocorrelation between fire and nofire patches. 

We can easily loop through all candidate pixels as each patch extraction removes quite many of those. In addition, we randomize this list in order not to have a bias towards fire "appearing" on the right.

> Note that there is still a bias in this type of dataset that the center pixel belongs to the fire class. Maybe analyze what happens if you try to remove this structure from the dataset.

After extracting all fire patches, we want to extract nofire patches of the same amount. Therefore, we practically use the same approach, but now the list of locations to look at is taken to be random and we check that each patch is completely zero. This approach -- with a longer loop of trying -- brings us quite a few patches and due to the random sampling we don't induce much neighboring patch structures. However, depending on the amount of fire, this approach does not always generate the same amount of nofire pixels, a fact that we can take care of later as the order of magnitude is okay in our dataset.

With these preparations, we have two directories of PNG images, one for fires and one for nofire patches and we have introduced a few hardcoded biases and at least one hyperparameter to watch and vary later: the patch size.

As we want to be able to use [pretrained models](#@pretrained), we will build a handful of 3-channel composites out of the available spectral channels - first in the classical combinations used for visualization, later maybe with a [data-driven sampling method](#@cite:igarss).

For now, we do this by a two-step process: first scaling a part of the spectral information to the [0,1] range for each of the source image channels and then building composites according to literature.


# A simple keras script for fine-tuning DNNs

This document is more about how to manage compute resources in ablation studies, where many models and configurations need to
be reliable processed, all results need to be transparently available, and complex evaluations across all runs need to be performed.

Therefore, we will be very short and dirty on the deep learning part, what we provide is kind-of okay, but there are lots of opportunities of improving the workload code in train.py

Nevertheless, the following aspects of train.py are powerful ideas that you should reflect. For example, the following block
is used to communicate configuration from job to script:
## Configuration and Reflection
```
import json
from types import SimpleNamespace

config = sys.argv[1]
cfg = json.load(open(config,"r"), object_hook=lambda d: SimpleNamespace(**d))
cfg.config=os.path.basename(config)
```
This means that the script does not have a lengthy definition of command line arguments like `--lr=0.01` for a learning rate.
Instead, all configuration is given as a JSON file. To further improve readability and writeability, we parse this into a
namespace such that we can access the fields with a `.` instead of brackets. This distinguishes it clearly from most data
we have. Further, we inject the configuration filename into the configuration and dump it right at the start of the script.
Should it fail or be a very good result, we can instantly see what has happened from the job's logfile without needing to look up the config file (which might have changed). This adds to full reproducibility as the correctness of the log does not depend on the configuration file not being lost or changed. The log itself is complete.

```
if __name__=="__main__":
    json_cfg = json.dumps({x:cfg.__dict__[x] for x in cfg.__dict__ if not x.startswith("_")})
    print("JSONCFG: %s" % (json_cfg))
```
By prefixing the JSON object with JSONCFG:, we can easily extract it using command line tools, more to it later.
## Seeting up DL model
The deep learning model setup follows
```
    conv = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=None,
                                             pooling=None,
                                             input_shape=cfg.img_input_shape)
 
    new_inputs = tf.keras.layers.Input(shape=cfg.img_input_shape)
    x = conv(new_inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg.hidden_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    new_outputs = tf.keras.layers.Dense(cfg.number_of_classes, activation='softmax')(x)

    model = tf.keras.Model(new_inputs, new_outputs)

    loss_fn = keras.losses.CategoricalCrossentropy() #from_logits=True
    optimizer = keras.optimizers.Adam(learning_rate = cfg.lr1)
    
    use_metrics=[]
    print(model.summary())
    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
```
There is nothing special here, no metrics during training helped with tensorflow stalling due to NaN losses.
With proper cluster setup that kill not only the job, but also the docker container in a reliable way, this can
certainly be reactivated to show progress during training beyond the loss itself.

For other models, based on configuration, only the `conv` variable or trainability need to be changed.

## Data Generation and Training Loop
We rely on the old-school ImageDataGenerator keras defaults for this script despite the fact that AtlasHDF or AtlasZARR provide
much better alternatives in most cases. Even TFrecords or other data formats can improve performance. But we stick to the
baseline in this document.

```
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0)
    train_generator = train_datagen.flow_from_directory(
    directory= "./%s/train/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    valid_generator = train_datagen.flow_from_directory(
        directory="./%s/val/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    test_generator = train_datagen.flow_from_directory(
        directory="./%s/test/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
    )
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    # Sometimes, your weights will go NaN (e.g., too large LR)
    # and it is not automatic that the job ends then. But we want to.
    
    callbacks = [ keras.callbacks.TerminateOnNaN() ]

    
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                        epochs=cfg.epochs,
                        callbacks = callbacks
    )
    model.save("%s.h5" % (cfg.config))

```
This creates a file with the same name as the configuration file name, but with h5 extension, for example, a file
`wf_swinfrared12-8a-4-0_0001-256.json` creates a weight file `wf_swinfrared12-8a-4-0_0001-256.json.h5`.
This can be used to see that training was successful in job scheduling.

The model is then evaluated on the test set quickly to have a performance indication in the log.
```
    use_metrics = [keras.metrics.Precision(),keras.metrics.Recall()]
    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
    
    STEPS_TEST=test_generator.n//test_generator.batch_size
    results = json.dumps(dict(zip(model.metrics_names, model.evaluate(test_generator, steps=STEPS_TEST))))
    print("TEST:%s" %(results))
```
Again, the performance on the test set is emitted as a JSON object beyond "TEST:" tag.

## Summary
This concludes the training script. Its general structure is to fail fast - no error handling on this level. Everything can be handled from the log files generated.

# A job script to run this across a cluster
The purpose of a job script is to start and manage the computation in limited time across a cluster of nodes. We use OpenPBS, an implementation of a PBS system with good configuration and performance.

## Job configuration
The script starts out with fixing a few variables

```
#!/bin/bash
#PBS -S /bin/bash
# This wants full node (full IO, full CPU, and full GPU)
#PBS -l select=1:ncpus=1:mem=10Gb
#PBS -l place=excl
#PBS -N trainlog
#PBS -l walltime=03:00:00
#PBS -j oe
```

This means essentially, that the job must have a complete node (assuming 1 GPU nodes, this is fine, refinement with requiring resources are
possible.



The following commented snippet shows how to come up with a sleeptime from the scheduler parameters itself. It is a bit hacky and does
only work if on all relevant nodes bc and qstat are installed and properly set up. Therefore, it is handy, but maybe there are many clusters
where it does not work out of the box. Then, just set SLEEPTIME yourself to a suitable number of seconds. Note that jobs are killed when
exceeding walltime PBS parameter and you might want to trigger a smooth shutdown (e.g., writing a checkpoint) yourself a little earlier. 
```
# walltime in seconds

#WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
#WALLTIME_SECONDS=$(echo "$WALLTIME" | sed -E 's/(.*):(.+):(.+)/\1*3600+\2*60+\3/;s/(.+):(.+)/\1*60+\2/' | bc)
#SLEEPTIME=$(($WALLTIME_SECONDS - 10))

# TODO: add PYTHONUNBUFFERED=1 to the environment to have smoother log processing especially for crashes on docker level
# like timeouts
```

Then, we output some useful information to the log file (start, host on which we are running if host-related problems appear, configuration file
name) before we spawn a docker conatiner.

```
SLEEPTIME=1800
echo "Spawning for $SLEEPTIME/${WALLTIME_SECONDS} seconds"

cd "$PBS_O_WORKDIR/"
echo  "START:"$(date)
echo "PWD: "$(pwd)
echo "HOST: "$(hostname)
echo "CONFIG: $CONFIG"
```
Spawning the docker is interesting as it contains a bit of magic. Let us go through all parameters to explain everything.
At the moment of invocation, we are on an NFS share, run a docker container with `-d` into the background. Further, we advise HDF5 not
to use file locking (as this is not efficient over NFS). We set the user ID to be the current job's user and group, we put the current
directory into the container at `/tf` and change our working directory there as well. We publish all GPUs, make it interactive, and delete it
if it is stopped. After that, we specify exactly a container by giving his SHA256 id. In this way, we can be almost sure that on all nodes the exact same software is run and not some `tensorflow:latest` which will be different if pulled on different times.


```
CONTAINER_ID=$(docker  run -d --env HDF5_USE_FILE_LOCKING=FALSE --user=$(id -u):$(id -g) -v $PWD:/tf -w /tf --gpus all -i --rm mwernerds/tf_example@sha256:5d8e6b7b315b0859b4a69abe51a1ea5dd4214132a217f995d28029051e3705bd\
		       python3 train.py $CONFIG)
```
The configuration variable `$CONFIG` contains the relative path of the configuration file, e.g., `configs/test.json` and can only be given via the environment as the PBS scheduler does not sensibly support job arguments. Everything is put into `CONTAINER_ID=$(...)`, hence, the output of the docker command is stored in the shell variable `CONTAINER_ID`. With the -d parameter, this is exactly the id of the container which we use later. Control is immediately given back to the job script such that we need to wait now for the job to fail or finnish.

## Waiting for the job

While waiting, we are gathering output from the container using `docker logs --follow` and spawn this into the background using the `&` at the end. In this way, most of the output within docker appears in our job logfile at the end. After that, we wait for `$SLEEPTIME` seconds, each time for 10 seconds using a bash for loop and the bash builtin command `seq`. Each time this loop is run, we check that the container is alive and running and if not, we terminate the job script, but without an error by `exit 0`. Otherwise, PBS would retry but it is very likely that the container exited successfully before the wallclock limit.

```
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
```
After this loop, we will have a running docker container, but no time left. That is, we kill the docker container with docker stop (it will remove itself as we have selected the `--rm` flag for it).

Note that it is essential that you remove the docker container before the job script is killed as it has no walltime anymore, otherwise, you might have zombie containers blocking GPU and all future jobs will fail as there is no GPU memory. This is why SLEEPTIME should be a bit smaller than the wallclock limit of PBS.

```
echo "KILL AT: $(date)"
docker stop "$CONTAINER_ID"
echo  "END:"$(date)
```
This completes the job script taking care of invocation of all services, collecting information and output and terminating early enough all resources.

# Job Submission System

The job submission is now rather easy, just a `qsub -v CONFIG=<name of config file> job.sh` will do it. qstat prints information on all jobs.
A small script to batch-submit all jobs that did not generate a weight file is provided with the warning that it will re-schedule all jobs that failed and maybe they will fail again for one or another reason. Therefore, it is important to delete job configurations that failed to avoid too much overhead.

The script is easy to read:
```
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
```
For all configuration files in configs/*.json, we check if the h5 file exists in one or another location. If not, we go for it.


# Job Configuration

A job configuration file looks like this:
```
{
  "dataset": "wf_swinfrared12-8a-4",
  "img_input_shape": [
    64,
    64,
    3
  ],
  "number_of_classes": 2,
  "lr1": 0.00001,
  "epochs": 1000,
  "dropout":0.1,
  "hidden_units": 128
}
```
This information is used to steer our train.py script to work on the right data with the right parameters.

Such scripts are easy to autogenerate with bash as the following examples show:
```
for dataset in wf_12-11-04 wf_agriculture11-8-2 wf_colorinfrared_8_4_3 wf_natural_4_3_2 wf_swinfrared12-8a-4; do
    for lr in 0.01 0.001 0.0001 1e-7; do
	for hidden in 32 64 128 256 512 1024; do
cat > $dataset-$(echo $lr |tr "." "_")-$hidden-50ep.conf <<EOF
{
"dataset": "$dataset",
"img_input_shape": [64,64,3],
"number_of_classes": 2,
"lr1": $lr,
"epochs":50,
"hidden_units": $hidden,
"dropout": 0.1
}
EOF
    done
done
done

```
The whole structure is a heredoc in which some places are replaced from for loop variables in bash. And a filename is generated (making sure it
has only one `.` with a tr comman invocation replacing `.` with `_` in the learning rate.

This can be run without the cluster, it just generates a bunch of files.

When running this, one gets a joblog for each of those and it contains a performance. We now could select one of the best configurations so far and train it for longer with a second script
```
for epochs in 10 50 100 200 500 1000; do
    for lr in 0.001 0.0001 0.00001 0.00001; do
	cat > best12-8a-$epochs-$(echo $lr |tr "." "_").json <<EOF

{
  "dataset": "wf_swinfrared12-8a-4",
  "img_input_shape": [
    64,
    64,
    3
  ],
  "number_of_classes": 2,
  "lr1": $lr,
  "epochs": $epochs,
  "dropout":0.1,
  "hidden_units": 128
}
EOF
    done
done
```
generating new jobs. Running `submit_all` will instruct all nodes to cooperate on transforming these as well.

# Peeking at the results

We are big fans of structured data and, therefore, rely on JQ to analyze the output of our programs. In order to do so,
we will scan all `trainlog` log files for reports. And each of those is parsed by removing the prefix and just output. Hence, we
get a stream of JSON objects which we can pipe to the JSON Query Processor jq. 

This procedure is stored in results.sh. In this script, we look at all trainlogs and if they don't contain results (e.g., they failed), we report a precision value of -1. This is done by extracting the configuration (based on JSONCFG) which is in all joblogs (remember that we output it as one of the first things). We extract the precision as the last line of the progress bars of the container. This is hacky and can be changed. But such a hack is an everyday help which is why we keep it inside. More concretely, the progress bars are unrolled by making every carriage return (which prepares the line for overwrite implementing the progress bar) with a newline and take the last line containing precision and parse for the precision itself by reversing, cutting and back-reversing.

Then, we patch the configuration with the logfile name and with the testprecision parsed.
```
#!/bin/bash

function loganalyze()
{
    #"echo  {}:; grep precision {} |tr '\r' '\n' | tail -1; cat {} |grep JSONCFG"
    if grep precision "$1" > /dev/null; then
	TESTPREC=$(grep precision "$1" |tr '\r' '\n' | tail -1 |rev | cut -d":" -f1  |rev  )
    else
	TESTPREC=-1
    fi
    grep JSONCFG "$1" | cut -d":" -f2- \
	| jq '. + {"logfile":"'"$1"'"}' \
	| jq '. + {"testprec":'"$TESTPREC"'}' 
	     
}

export -f loganalyze


ls trainlog*  | parallel loganalyze {}

```
Parallel takes care of running this. Be careful not to run it over the cluster, this is not intended.

# Result analysis

The results can now be peeked at and we just give two queries which we use a lot. You should maybe learn more about JQ to
be able to answer your queries.

## A simple histogram of precision
```
martin@martin:/data/share/tf2_cluster_example$ ./results.sh  | jq .testprec | cut -b 1-4 |sort -g |uniq -c
    166 -1
     33 0.50
      2 0.60
      2 0.62
      1 0.63
      1 0.64
      1 0.69
      5 0.70
      6 0.71
      4 0.72
      3 0.73
      1 0.74
      3 0.78
      2 0.79
      1 0.80
      2 0.81
      1 0.82
     11 0.83
     10 0.84
      8 0.85
      8 0.86
      2 0.87
      2 0.88

```
Here we see that 33 models did not converge (too large learning rate or whatever), 166 jobs failed (e.g., NaN?, erroneous files in the dataset?)
and there is a clear cluster of 20 models having 83% to 84% precision.

We can also easily query for the best model
```
martin@martin:/data/share/tf2_cluster_example$ ./results.sh | jq -s 'max_by(.testprec)'
{
  "dataset": "wf_swinfrared12-8a-4",
  "img_input_shape": [
    64,
    64,
    3
  ],
  "number_of_classes": 2,
  "lr1": 0.0001,
  "epochs": 10,
  "hidden_units": 128,
  "dropout": 0.1,
  "config": "wf_swinfrared12-8a-4-0_0001-128.json",
  "logfile": "trainlog.o663",
  "testprec": 0.881
}
```

Or for certain aspects of certain ranges:
```
martin@martin:/data/share/tf2_cluster_example$ ./results.sh | jq 'select(.dataset=="wf_natural_4_3_2") | .testprec' |cut -b 1-4 |sort |uniq -c
    113 -1
```
In fact this shows that most likely the dataset natural is broken (we added a few empty images to it). Now trying to find the reason with JQ is easy as well, let us take one of these records

```
martin@martin:/data/share/tf2_cluster_example$ ./results.sh | jq -r  'select(.dataset=="wf_natural_4_3_2") | select (.testprec < 0) | .logfile' 
| sort -R | head  -1                                                                                                                           
trainlog.o736
```
This gives us a random job logfile with these parameters we can now just inspect. The given file ends with following error information
from within the container:
```
STATUS: Container died roughly after 10 seconds. Exiting assuming succes
tail: '20' kann nicht zum Lesen geöffnet werden: Datei oder Verzeichnis nicht gefunden
martin@martin:/data/share/tf2_cluster_example$ cat trainlog.o736 |tail -20
  File "/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/utils.py", line 110, in load_img
    img = pil_image.open(path)

  File "/usr/local/lib/python3.6/dist-packages/PIL/Image.py", line 3031, in open
    "cannot identify image file %r" % (filename if filename else fp)

PIL.UnidentifiedImageError: cannot identify image file './wf_natural_4_3_2/train/fire/fire-S2A_20210926T185131_00_A_170e999b-273c-4c5d-a8c3-25ef
2096d034_prod-29-4633.png'                                                                                                                     


         [[{{node PyFunc}}]]
         [[IteratorGetNext]]
         [[IteratorGetNext/_2]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_function_1934]

Function call stack:
train_function -> train_function

Inspecting after 10
STATUS: Container died roughly after 10 seconds. Exiting assuming succes
martin@martin:/data/share/tf2_cluster_example$
```

You can now continue to find such problems by grepping directly across all joblogs for UnidentifiedImageError...

Or you just repair the dataset.


# Container Build

The last step is to set up a sensible docker container for our system. As we are not running as root, we cannot install software on the cluster
while the job is running - not even within the docker container (at least not without setting up a home directory for the user and lots of other stuff). Therefore, we just build and push a container.

Again, we fix the exact version of the upstream container for reproducibility. Note that this just implies that the drivers need to be new enough as most of NVIDIAs hardware stack is backward compatible.

The Dockerfile reads like this: 
```
FROM tensorflow/tensorflow@sha256:3f8f06cdfbc09c54568f191bbc54419b348ecc08dc5e031a53c22c6bba0a252e
RUN pip3 install Pillow;
```
We realized that we only need Pillow, hence, we did not put a requirements.txt file somewhere.

We add a Makefile to make sure that container names are given (why aren't container names inside Dockerfiles?).
It looks like

```
all:
	docker build . -t mwernerds/tf_example
```

Note that you have to push the image to containers and use the SHA256 hash in your job script to wire everything correctly.

# Appendix
The following scripts are from git 12568fee0fda4bf7d526e7e7573e1528f527d762 taken on 2022-04-19.
The git itself is not yet published.

## Job Script
```
#!/bin/bash
#PBS -S /bin/bash
# This wants full node (full IO, full CPU, and full GPU)
#PBS -l select=1:ncpus=1:mem=10Gb
#PBS -l place=excl
#PBS -N trainlog
#PBS -l walltime=03:00:00
#PBS -j oe

# walltime in seconds

#WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
#WALLTIME_SECONDS=$(echo "$WALLTIME" | sed -E 's/(.*):(.+):(.+)/\1*3600+\2*60+\3/;s/(.+):(.+)/\1*60+\2/' | bc)
#SLEEPTIME=$(($WALLTIME_SECONDS - 10))

# TODO: add PYTHONUNBUFFERED=1 to the environment to have smoother log processing especially for crashes on docker level
# like timeouts

SLEEPTIME=1800
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
```

## Train.py
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from numpy import dot
from numpy.linalg import norm
import os
import sys
import numpy as np

import json
from types import SimpleNamespace

config = sys.argv[1]
cfg = json.load(open(config,"r"), object_hook=lambda d: SimpleNamespace(**d))
cfg.config=os.path.basename(config)

    
    
if __name__=="__main__":
    json_cfg = json.dumps({x:cfg.__dict__[x] for x in cfg.__dict__ if not x.startswith("_")})
    print("JSONCFG: %s" % (json_cfg))
    conv = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=None,
                                             pooling=None,
                                             input_shape=cfg.img_input_shape)
 
    new_inputs = tf.keras.layers.Input(shape=cfg.img_input_shape)
    x = conv(new_inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg.hidden_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    new_outputs = tf.keras.layers.Dense(cfg.number_of_classes, activation='softmax')(x)

    model = tf.keras.Model(new_inputs, new_outputs)

    loss_fn = keras.losses.CategoricalCrossentropy() #from_logits=True
    optimizer = keras.optimizers.Adam(learning_rate = cfg.lr1)
    
    use_metrics=[]
    print(model.summary())
    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
    # data

        
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0)
    train_generator = train_datagen.flow_from_directory(
    directory= "./%s/train/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    valid_generator = train_datagen.flow_from_directory(
        directory="./%s/val/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    test_generator = train_datagen.flow_from_directory(
        directory="./%s/test/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
    )
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    # Sometimes, your weights will go NaN (e.g., too large LR)
    # and it is not automatic that the job ends then. But we want to.
    
    callbacks = [ keras.callbacks.TerminateOnNaN() ]

    
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                        epochs=cfg.epochs,
                        callbacks = callbacks
    )
    model.save("%s.h5" % (cfg.config))
    # Evaluate on Testset
    use_metrics = [keras.metrics.Precision(),keras.metrics.Recall()]

    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
    
    STEPS_TEST=test_generator.n//test_generator.batch_size
    results = json.dumps(dict(zip(model.metrics_names, model.evaluate(test_generator, steps=STEPS_TEST))))
    print("TEST:%s" %(results))
```

## Container Build
```
martin@martin:/data/share/tf2_cluster_example/container_build$ make
docker build . -t mwernerds/tf_example
Sending build context to Docker daemon  3.072kB
Step 1/2 : FROM tensorflow/tensorflow@sha256:3f8f06cdfbc09c54568f191bbc54419b348ecc08dc5e031a53c22c6bba0a252e
 ---> f5ba7a196d56
Step 2/2 : RUN pip3 install Pillow;
 ---> Running in bae605e91888
Collecting Pillow
  Downloading Pillow-8.4.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
Installing collected packages: Pillow
Successfully installed Pillow-8.4.0
WARNING: You are using pip version 20.1; however, version 21.3.1 is available.
You should consider upgrading via the '/usr/bin/python3 -m pip install --upgrade pip' command.
Removing intermediate container bae605e91888
 ---> e342fc7fa699
Successfully built e342fc7fa699
Successfully tagged mwernerds/tf_example:latest
martin@martin:/data/share/tf2_cluster_example/container_build$ docker push mwernerds/tf_example
Using default tag: latest
The push refers to repository [docker.io/mwernerds/tf_example]
06864633d393: Pushed 
befce55f84c6: Mounted from tensorflow/tensorflow 
d870b0f92c14: Mounted from tensorflow/tensorflow 
2c86cf4cd527: Mounted from tensorflow/tensorflow 
1cccc1f74e01: Mounted from tensorflow/tensorflow 
20119e4b0fc9: Mounted from tensorflow/tensorflow 
751ae3b79e0a: Mounted from tensorflow/tensorflow 
133ee43735a0: Mounted from tensorflow/tensorflow 
97c83918ca41: Mounted from tensorflow/tensorflow 
6b87768f66a4: Mounted from tensorflow/tensorflow 
808fd332a58a: Mounted from tensorflow/tensorflow 
b16af11cbf29: Mounted from tensorflow/tensorflow 
37b9a4b22186: Mounted from tensorflow/tensorflow 
e0b3afb09dc3: Mounted from tensorflow/tensorflow 
6c01b5a53aac: Mounted from tensorflow/tensorflow 
2c6ac8e5063e: Mounted from tensorflow/tensorflow 
cc967c529ced: Mounted from tensorflow/tensorflow 
latest: digest: sha256:5d8e6b7b315b0859b4a69abe51a1ea5dd4214132a217f995d28029051e3705bd size: 3886
martin@martin:/data/share/tf2_cluster_example/container_build$
```