# Rationale
In order to use non-standard software, you need to build a container yourself as it will run in non-privileged mode in our infrastructure.

Here is one example extending a quite old version of tensorflow (check the SHA hash) that currently works with our hardware-oriented projects
with Pillow to allow for image loading.

- build the image
- push it to the global repo dockerhub (later our own)
- change the job file to reflect the repo


# Appendix



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
martin@martin:/data/share/tf2_cluster_example/container_build$ docker push
"docker push" requires exactly 1 argument.
See 'docker push --help'.

Usage:  docker push [OPTIONS] NAME[:TAG]

Push an image or a repository to a registry
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