#!/bin/bash
docker build -t arkangelia/app_face_reco:latest .

docker run -it \
    --rm \
    --net="host"\
    --name="app_face_reco"\
    -v $(pwd)/..:/app_face_reco \
    arkangelia/app_face_reco:latest