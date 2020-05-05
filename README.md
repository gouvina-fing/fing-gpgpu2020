# GPGPU 2020 (FING)
Tareas del curso de Computación de Propósito General en Unidades de Procesamiento Gráfico 2020, Facultad de Ingeniería (UdelaR)
- **Práctico 1 - Impacto del acceso a los datos en CPU**
- **Práctico 2 - Primeros pasos con CUDA**
- **Práctico 3 - Blur y ajustar brillo en imágenes**

## Setup Environment

### Prerequisites
- Recommended Drivers of your NVIDIA video card: `sudo ubuntu-drivers autoinstall`
- Nvidia Container Runtime:
    - Add the repository [following the instructions](https://nvidia.github.io/nvidia-container-runtime/)
    - After doing `sudo apt-get update`, install the Container Runtime: `sudo apt-get install nvidia-container-runtime`

### Build the image and name it gpgpu
`docker image build -t gpgpu .`

### Run the container
`docker run -it --gpus all --network host -v $(pwd):/workspace/src gpgpu`

### Remove dangling images
`docker rmi -f $(docker images -f "dangling=true" -q)`


## Cluster

### Connect
`ssh gpgpu_6@login.cluster.uy`

### Add volume (Linux)
- Open `Files`
- Click on `+ Other Locations`
- Connect to Server inputing the address: `ssh://gpgpu_6@login.cluster.uy/clusteruy/home/gpgpu_6`

You may relocate proyect files to this new volume in order to access them on the cluster

### Run job
- Alter `launch_single.sh` to your liking
- `sbatch launch_single.sh`
- You can also swap the last line for `$1 $2 $3` and run `sbatch launch_single.sh ./program arg1 arg2` (in case enqueuing with multiple scripts/args is desired)