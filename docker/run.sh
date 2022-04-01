#!/usr/bin/env bash

IMAGE=ghcr.io/askiggy/iggy-enrich-demos:latest

if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
  echo "Image not found, building from recipe...."
  cp iggy-metaflow-demo/requirements.txt docker
  cd docker
  docker build . --no-cache --rm -t $IMAGE
  rm requirements.txt
  cd ..
fi

docker run --rm -it \
	--name iggy-enrich-demos \
	-v $(pwd):/content \
	-w /content \
	$IMAGE
