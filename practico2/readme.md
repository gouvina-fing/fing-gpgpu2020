# Práctico 2

## Prerequisites
- Recommended Drivers of your NVIDIA video card: `sudo ubuntu-drivers autoinstall`
- Nvidia Container Runtime: `sudo apt-get install nvidia-container-runtime`

## Build the image and name it gpgpu
`docker image build -t gpgpu .`

## Run the container
`docker run -it --gpus all --network host -v $(pwd):/workspace/src gpgpu`

## Remove dangling images
`docker rmi -f $(docker images -f "dangling=true" -q)`
