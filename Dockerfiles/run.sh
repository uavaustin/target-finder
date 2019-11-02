xhost +"local:docker@"

if [ $# -eq 0 ] ; then
  sudo docker run --runtime=nvidia -ti \
    uavaustin/target-finder-model-env:tf /bin/bash
else 
  sudo docker run --runtime=nvidia -ti \
  -v $1:/target-finder \
  uavaustin/target-finder-model-env:tf /bin/bash

fi
