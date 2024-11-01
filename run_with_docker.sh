#!/bin/bash

# This is to determine Docker version for the command
version_less_than_equal_to() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" = "$1"; }

REQ_DOCKER_VERSION=19.03
docker_version=$(docker -v | cut -d ' ' -f3 | sed 's/,$//')

if version_less_than_equal_to $REQ_DOCKER_VERSION $docker_version; then
    # Use the normal docker command
    DOCKER_CMD="docker run --rm --runtime=nvidia --gpus all"
else
    # Use nvidia-docker
    DOCKER_CMD="nvidia-docker run --runtime=nvidia"
fi


# this block is for running X apps in docker
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi




# now, let's mount the user directory into the Docker container
# set the environment varible SDL_VIDEODRIVER to SDL_VIDEODRIVER_VALUE


UNREAL_ROOT_PATH=/home/daniel/Docker/AirSim
AirSim_Setting=/home/daniel/Docker/AirSim/docker/settings.json


sudo docker run --runtime=nvidia --gpus all -it \
    -v ${AirSim_Setting}:/home/airsim_user/Documents/AirSim/settings.json \
    -v $UNREAL_ROOT_PATH:$UNREAL_ROOT_PATH \
    -e SDL_VIDEODRIVER='' \
    -e SDL_HINT_CUDA_DEVICE='0' \
    --net=host \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=/tmp/.docker.xauth" \
    --volume="/tmp/.docker.xauth:/tmp/.docker.xauth" \
    --rm \
    jinhuiye/airsim_binary:1.2 \
    bash -c "cd $UNREAL_ROOT_PATH && /bin/bash"
    
    

# run in your teminal
# bash path_to/Blocks/Blocks.sh -windowed -ResX=1080 -ResY=720

