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
