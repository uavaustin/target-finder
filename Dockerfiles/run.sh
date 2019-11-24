xhost +"local:docker@"

sudo docker run --runtime=nvidia -ti \
-v $PWD/../:/host/mounted \
uavaustin/target-finder-model-env:tf1 /bin/bash
