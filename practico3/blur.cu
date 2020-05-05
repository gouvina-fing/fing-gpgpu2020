#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

// CUDA Thread Indexing Cheatsheet https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf

// Ej 2a) Kernel que aplica el filtro Gaussiano en la GPU
// Ejemplo filtro https://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf
// Ejemplo multiplicacion de matrices http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html
__global__ void blur_kernel(float* d_input, int width, int height, float* d_output, float* d_msk, int m_size) {
    
    // __shared__ float block_memory[1024];

    int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    imgx = max(0, imgx);
    imgx = min(imgx, width - 1);

    int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;
    imgy = max(0, imgy);
    imgy = min(imgy, height - 1);

    //int block_index = (threadIdx.y * blockDim.y) + threadIdx.x;
    //block_memory[block_index] = d_input[(imgy*width) + imgx];
    
    __syncthreads();

    float val_pixel = 0;

    // Aca aplicamos la mascara
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < m_size; j++) {
            
            int i2 = i - m_size/2;
            int j2 = j - m_size/2;

            int ix = imgx + i2;
            int iy = imgy + j2;
            
            //int bindex = ((threadIdx.y + i2) * blockDim.y) + (threadIdx.x + j2);

            // Altera el valor de un pixel, según sus vecinos.
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) { // && bindex >= 0 && bindex < 1024) {
                //val_pixel = val_pixel +  block_memory[bindex] * d_msk[i*m_size+j];
                val_pixel = val_pixel +  d_input[(iy*width) + ix] * d_msk[i*m_size+j];
            }
        }
    }

    
    if (imgx < width && imgy < height) {
        d_output[(imgy*width) + imgx] = min(255.0f, max(0.0f, val_pixel));
    }
}

// Ej 1a) Threads con índice consecutivo en la dirección x deben acceder a pixels de una misma fila de la imagen.
//        Es importante usar blockIdx.x, blockIdx.y, threadIdx.x y threadIdx.y adecuadamente para acceder a la estructura bidimensional.
__global__ void ajustar_brillo_coalesced_kernel(float* d_input, float* d_output, int width, int height, float coef) {
    int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (imgx < width && imgy < height) {
        d_output[(imgy*width) + imgx] = min(255.0f, max(0.0f, d_input[(imgy*width) + imgx] + coef));
    }
}

// Ej 1a) Threads con índice consecutivo en la dirección x deben acceder a pixels de una misma columna de la imagen.
//        Es importante usar blockIdx y threadIdx adecuadamente para acceder a la estructura bidimensional.
__global__ void ajustar_brillo_no_coalesced_kernel(float* d_input, float* d_output, int width, int height, float coef) {
    int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (imgx < width && imgy < height) {
        d_output[(imgx*height) + imgy] = min(255.0f, max(0.0f, d_input[(imgx*height) + imgy] + coef));
    }
}

// Procesa la img en GPU sumando un coeficiente entre -255 y 255 a cada píxel, aumentando o reduciendo su brillo.
void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int algorithm, int filas=1) {

    // Tamaño de img_in en memoria
    unsigned int size = width * height * sizeof(float);
    float * device_img_in = (float *)malloc(size);
    float * device_img_out = (float *)malloc(size);

    // Reservo memoria en la GPU
    CUDA_CHK(cudaMalloc((void**)& device_img_in, size));
    CUDA_CHK(cudaMalloc((void**)& device_img_out, size));

    // Copio los datos a la memoria de la GPU
    CUDA_CHK(cudaMemcpy(device_img_in, img_in, size, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
   
    // Configurar grilla y lanzar kernel
    // TODO: La grilla (bidimensional) de threads debe estar configurada para aceptar matrices de cualquier tamaño.
    int block_size = 32;
    int block_amount_x = width / block_size + (width % block_size != 0); // Division with ceiling
    int block_amount_y = height / block_size + (height % block_size != 0); // Division with ceiling

    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(block_size, block_size); // Block dimension

    // Lanzar kernel
    CLK_CUEVTS_INIT;
    CLK_CUEVTS_START;

    switch(algorithm) {
        case 1:
            ajustar_brillo_coalesced_kernel<<<tamGrid, tamBlock>>>(device_img_in, device_img_out, width, height, coef);
            break;
        case 2:
            ajustar_brillo_no_coalesced_kernel<<<tamGrid, tamBlock>>>(device_img_in, device_img_out, width, height, coef);
            break;
    }
    cudaDeviceSynchronize();

    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;

    printf("Tiempo ajustar brillo GPU: %f ms\n", t_elap);

    // Transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(img_out, device_img_out, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia

    // TODO: Ej 1b) Registrar tiempos de cada etapa de ajustar_brillo_gpu para las dos variantes. Discutir diferencia entre variantes.
    //              (tiempos, reserva de memoria, transferencia de datos, ejecución del kernel, etc)
    //              Usar ambos mecanismo de medidas de utils.h (deberian dar casi igual)

    // TODO: Ej 1c) Compare los resultados de la salidad de nvprof.
    //              Registrar con nvprof --profileapi-trace none --metrics gld_efficiency ./blur imagen.ppm
    //              Qué puede decir del resultado de la métrica gld_efficiency?
    //              Duda: Esto se hace acá o en main.cpp?

    // Libero la memoria en la GPU
    CUDA_CHK(cudaFree(device_img_in));
    CUDA_CHK(cudaFree(device_img_out));
}

// Ej 2) Aplica un filtro Gaussiano que reduce el ruido de una imagen en escala de grises.
//       El filtro sustituye el valor de intensidad de cada pixel por un promedio ponderado de los pixeles vecinos.
//       Los pesos por los cuales se pondera cada vecino en el promedio se almacenan en una matriz cuadrada (máscara)
void blur_gpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){
    
    // Tamaño de img_in en memoria
    unsigned int size = width * height * sizeof(float);
    float * device_img_in = (float *)malloc(size);
    float * device_img_out = (float *)malloc(size);

    // Reservo memoria en la GPU
    CUDA_CHK(cudaMalloc((void**)& device_img_in, size));
    CUDA_CHK(cudaMalloc((void**)& device_img_out, size));

    // Copio los datos a la memoria de la GPU
    CUDA_CHK(cudaMemcpy(device_img_in, img_in, size, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
   
    // Configurar grilla y lanzar kernel
    // TODO: La grilla (bidimensional) de threads debe estar configurada para aceptar matrices de cualquier tamaño.
    int block_size = 32;
    int block_amount_x = width / block_size + (width % block_size != 0); // Division with ceiling
    int block_amount_y = height / block_size + (height % block_size != 0); // Division with ceiling

    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(block_size, block_size); // Block dimension

    CLK_CUEVTS_INIT;
    CLK_CUEVTS_START;

    blur_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out, msk, m_size);
    cudaDeviceSynchronize();

    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;

    printf("Tiempo filtro gaussiano GPU: %f ms\n", t_elap);

    // Transferir resultado a la memoria principal
    CUDA_CHK(cudaMemcpy(img_out, device_img_out, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia

    // TODO: Ej 2b) Registre los tiempos de cada etapa de la función y compare las variantes de CPU y GPU.
    //              Usar ambos mecanismo de medidas de utils.h (deberian dar casi igual)
    //              ¿Qué aceleración se logra? ¿Y considerando únicamente el tiempo del kernel (cudaMemcpy tiene mucho overhead!)?
    //              Duda: Esto se hace acá o en main.cpp?

    // Libero la memoria en la GPU
    CUDA_CHK(cudaFree(device_img_in));
    CUDA_CHK(cudaFree(device_img_out));
}

// Recorre la imagen sumando secuencialmente un coeficiente entre -255 y 255 a cada píxel, aumentando o reduciendo su brillo.
void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef) {

    CLK_POSIX_INIT;
    CLK_POSIX_START;

    for(int imgx=0; imgx < width ; imgx++) {
        for(int imgy=0; imgy < height; imgy++) {
            img_out[imgy*width+imgx] = min(255.0f,max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    printf("Tiempo ajustar brillo CPU: %f ms\n", t_elap);
}

// Recorre la imagen aplicando secuencialmente un filtro Gaussiano que reduce el ruido de una imagen en escala de grises.
void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size) {

    CLK_POSIX_INIT;
    CLK_POSIX_START;

    float val_pixel=0;
    
    // Para cada pixel aplicamos el filtro
    for(int imgx=0; imgx < width ; imgx++) {
        for(int imgy=0; imgy < height; imgy++) {

            val_pixel = 0;

            // Aca aplicamos la mascara
            for (int i = 0; i < m_size ; i++) {
                for (int j = 0; j < m_size ; j++) {
                    
                    int ix =imgx + i - m_size/2;
                    int iy =imgy + j - m_size/2;
                    
                    // Altera el valor de un pixel, según sus vecinos.
                    if(ix >= 0 && ix < width && iy>= 0 && iy < height)
                        val_pixel = val_pixel +  img_in[iy * width +ix] * msk[i*m_size+j];
                }
            }
            
            // Guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    printf("Tiempo filtro Gaussiano CPU: %f ms\n", t_elap);
}