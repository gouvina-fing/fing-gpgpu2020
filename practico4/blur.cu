#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

// CUDA Thread Indexing Cheatsheet https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
// Ejemplo filtro https://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf
// Ejemplo multiplicacion de matrices http://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html

// Ej 2a) Kernel que aplica el filtro Gaussiano en la GPU empleando memoria compartida (comprarar tiempos y nvprof con practico3 blur sin mascara const float* __restrict__ d_msk)
// Ej 2b-1) Agregar máscara const float* __restrict__ d_msk (y comparar tiempos con 2a)
//          Estas flags dicen que: el dato es de solo lectura (const) y es la unica versión de ese puntero (__restrict__)
//          Permite al compilador hacer optimizaciones y usar la cache constante
// Ej 2b-2) Copiar máscara con __constant__ y cudaMemcpyToSymbol (para que resida en mem constante) (y comparar tiempos con 2b-1)
//          Acá estamos optimizando la memoria constante.
//          La memoria constante es de 64KB, está optimizada para que si todo acceso del warp accede al mismo elem el acceso es óptimo
__global__ void blur_kernel(float* d_input, int width, int height, float* d_output, float* d_msk, int m_size, int shared_mem_height, int shared_mem_size) {

    __shared__ float block_memory[shared_mem_size];
    
    int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;
    int radius = m_size/2;

    // Carga de la memoria compartida maximizando paralelismo entre hilos
    
    int shifted_imgx = imgx - radius;
    int shifted_imgy = imgy - radius;
    int right_shifted_imgx = imgx - radius + blockDim.x;
    int under_shifted_imgy = imgy - radius + blockDim.y;

    int right_shifted_memory_index_x = threadIdx.x + blockDim.x;
    int under_shifted_memory_index_y = threadIdx.y + blockDim.y;

    int memory_position_y = threadIdx.y*shared_mem_height
    int under_shifted_memory_position_y = under_shifted_memory_index_y*shared_mem_height
    
    // Cada hilo carga su lugar shifteado 2 posiciones hacia la izquierda y 2 hacia arriba (-2, -2)
    
    if (shifted_imgx >= 0 && shifted_imgx < width && shifted_imgy >= 0 && shifted_imgy < height) {
        block_memory[memory_position_y + threadIdx.x] = d_input[(shifted_imgy*width) + shifted_imgx]
    }
    
    // Cada hilo carga su lugar shifteado (blockDim.x - 2) posiciones hacia la derecha y 2 hacia arriba (+29, -2)
    if (right_shifted_imgx >= 0 && right_shifted_imgx < width && shifted_imgy >= 0 && shifted_imgy < height) {
        block_memory[memory_position_y + right_shifted_memory_index_x] = d_input[(shifted_imgy*width) + right_shifted_imgx]
    }

    // Cada hilo carga su lugar shifteado 2 posiciones hacia la izquierda y (blockDim.y - 2) hacia abajo (-2, +29)
    if (shifted_imgx >= 0 && shifted_imgx < width && under_shifted_imgy >= 0 && under_shifted_imgy < height) {
        block_memory[under_shifted_memory_position_y + threadIdx.x] = d_input[(under_shifted_imgy*width) + shifted_imgx]
    }

    // Cada hilo carga su lugar shifteado (blockDim.x - 2) posiciones hacia la derecha y (blockDim.y - 2) hacia abajo (+29, +29)
    if (right_shifted_imgx >= 0 && right_shifted_imgx < width && under_shifted_imgy >= 0 && under_shifted_imgy < height) {
        block_memory[under_shifted_memory_position_y + right_shifted_memory_index_x] = d_input[(under_shifted_imgy*width) + right_shifted_imgx]
    }

    __syncthreads();
    
    float val_pixel = 0;

    // Aca aplicamos la máscara
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < m_size; j++) {

            int ix = imgx + i - m_size / 2;
            int iy = imgy + j - m_size / 2;
            
            // Altera el valor de un pixel, según sus vecinos.
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                val_pixel = val_pixel + d_input[(iy * width) + ix] * d_msk[i*m_size+j];
            }

            /*
                Versión memoria compartida:
                
                int bindex = block_index + (iy * blockDim.x) + ix; //((threadIdx.y + i - m_size / 2) * blockDim.y) + (threadIdx.x + j - m_size / 2);
                
                if (bindex >= 0 && bindex < 1024) {
                    val_pixel = val_pixel +  block_memory[bindex] * d_msk[i*m_size+j];
                }
                else if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    val_pixel = val_pixel + d_input[(iy * width) + ix] * d_msk[i*m_size+j];
                }
            */
        }
    }
    
    if (imgx < width && imgy < height) {
        d_output[(imgy*width) + imgx] = val_pixel;
    }
}

// Ej 2) Aplica un filtro Gaussiano que reduce el ruido de una imagen en escala de grises.
//       El filtro sustituye el valor de intensidad de cada pixel por un promedio ponderado de los pixeles vecinos.
//       Los pesos por los cuales se pondera cada vecino en el promedio se almacenan en una matriz cuadrada (máscara)
void blur_gpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){
    
    // Auxiliar para contar tiempo total
    float t_total = 0;
    
    // Etapa 1: Reserva de Memoria
    CLK_CUEVTS_INIT;
    CLK_CUEVTS_START;

    // Reserva en CPU
    unsigned int size = width * height * sizeof(float);
    unsigned int size_msk = 25 * sizeof(float);
    float * device_img_in = (float *)malloc(size);
    float * device_img_out = (float *)malloc(size);
    float * device_msk = (float *)malloc(size_msk);

    // Reserva en GPU
    CUDA_CHK(cudaMalloc((void**)& device_img_in, size));
    CUDA_CHK(cudaMalloc((void**)& device_img_out, size));
    CUDA_CHK(cudaMalloc((void**)& device_msk, size_msk));
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo filtro gaussiano GPU (Reserva de memoria): %f ms\n", t_elap);
    t_total = t_total + t_elap;
    
    // Etapa 2: Transferencia de datos (Host -> Device)
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(device_img_in, img_in, size, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CUDA_CHK(cudaMemcpy(device_msk, msk, size_msk, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo filtro gaussiano GPU (Transferencia de datos (Host -> Device)): %f ms\n", t_elap);
    t_total = t_total + t_elap;

    // Etapa 3: Definir grilla
    int block_size = 32; // TODO: Definir constante
    int block_amount_x = width / block_size + (width % block_size != 0); // Division with ceiling
    int block_amount_y = height / block_size + (height % block_size != 0); // Division with ceiling
    int block_shared_mem_height = (block_size + m_size/2)
    int block_shared_mem_size = block_shared_mem_height*block_shared_mem_height
    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(block_size, block_size); // Block dimension

    // Etapa 4 : Lanzar Kernel
    CLK_CUEVTS_START;
    blur_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out, device_msk, m_size, block_shared_mem_height, block_shared_mem_size);
    // Sincronizar threads antes de parar timers
    cudaDeviceSynchronize(); 
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo filtro gaussiano GPU (Kernel): %f ms\n", t_elap);
    t_total = t_total + t_elap;

    // Etapa 5: Transferencia de Datos (Device -> Host)
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(img_out, device_img_out, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo filtro gaussiano GPU (Transferencia de datos (Host <- Device)): %f ms\n", t_elap);
    t_total = t_total + t_elap;
    printf("Tiempo filtro gaussiano GPU: %f ms\n", t_total);
    printf("\n");

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_img_in));
    CUDA_CHK(cudaFree(device_img_out));
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
    printf("\n");
}