#!/bin/bash
docker run -it --rm \
 -p 8888:8888 \
 -e DISPLAY=host.docker.internal:0 \
 -v $HOME/.Xauthority:/root/.Xauthority:rw \
 -v /mnt/c:/mnt/c:rw \
 -v /home/yammouch:/home/yammouch:rw \
 --entrypoint bash \
 fenics0010
# -p 8889:8888 \
# -u `id -u`:`id -g` \
# -v $HOME:$HOME:rw \
# ubuntux
