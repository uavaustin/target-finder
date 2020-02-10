sudo docker run --runtime=nvidia -ti \
-v $PWD:/host \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
uavaustin/target-finder-env:latest /bin/bash
