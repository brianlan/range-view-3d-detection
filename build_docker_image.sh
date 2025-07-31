docker run -it --rm --privileged --gpus all --name buildkit \
  -v $(pwd):/workspace \
  -v ~/.docker:/root/.docker \
  -v buildkit-data-vol:/var/lib/buildkit \
  --add-host=host.docker.internal:172.17.0.1 \
  -e http_proxy=http://host.docker.internal:20171 \
  -e https_proxy=http://host.docker.internal:20171 \
  -e HTTP_PROXY=http://host.docker.internal:20171 \
  -e HTTPS_PROXY=http://host.docker.internal:20171 \
  -w /workspace \
  --entrypoint buildctl-daemonless.sh \
  crazymax/buildkit:v0.23.2-ubuntu-nvidia \
  build \
    --frontend dockerfile.v0 \
    --local context=/workspace \
    --local dockerfile=/workspace/dockerfiles/rv3ddet \
    --output type=image,name=docker.io/brianlan/rv3ddet:v1,push=true
