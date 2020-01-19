xhost +"local:docker@"

sudo docker run --runtime=nvidia -ti \
-v $PWD/../:/host/mounted \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
uavaustin/target-finder-model-env:tf1 /bin/bash
