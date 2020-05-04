FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update

# Specify workspace
WORKDIR /workspace/src/

# If a build.sh needs to be executed uncomment:
COPY build.sh /workspace/src/build.sh
RUN ./build.sh

ENTRYPOINT [ "/bin/bash" ]
