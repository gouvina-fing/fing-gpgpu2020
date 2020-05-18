#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

#define TILE_WIDTH    32
#define TILE_HEIGHT   32
#define BLOCK_WIDTH   32
#define BLOCK_HEIGHT  32

// Ej 1a) Kernel 
__global__ void transpose_global_kernel(float* d_input, int width, int height, float* d_output) {
    unsigned int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (imgx < width && imgy < height) {
        d_output[(imgx*height) + imgy] = d_input[(imgy*width) + imgx];
    }
}

// Ej 1b) Kernel 
__global__ void transpose_shared_kernel(float* d_input, int width, int height, float* d_output) {
    
    __shared__ float tile[TILE_WIDTH][TILE_HEIGHT];

    // Indices (x,y) en imagen de entrada
    unsigned int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Lectura por fila (global) / Escritura por fila (compartida)
    if (imgx < width && imgy < height ) {
        tile[threadIdx.y][threadIdx.x] = d_input[(imgy*width) + imgx];
        __syncthreads();
    }

    // Indices (x,y) en imagen de salida ((y,x) en imagen de entrada)
    imgx = (blockIdx.y * blockDim.x) + threadIdx.x;
    imgy = (blockIdx.x * blockDim.y) + threadIdx.y;

    // Lectura por columna (compartida) / Escritura por fila (global)
    if (imgx < height && imgy < width) {
        d_output[(imgy*height) + imgx] = tile[threadIdx.x][threadIdx.y] ;
    }
}

// Ej 1c) Kernel 
__global__ void transpose_shared_extra_kernel(float* d_input, int width, int height, float* d_output) {

    __shared__ float tile[TILE_WIDTH][TILE_HEIGHT + 1];

    // Indices (x,y) en imagen de entrada
    unsigned int imgx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int imgy = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Lectura por fila (global) / Escritura por fila (compartida)
    if (imgx < width && imgy < height ) {
        tile[threadIdx.y][threadIdx.x] = d_input[(imgy*width) + imgx];
        __syncthreads();
    }

    // Indices (x,y) en imagen de salida ((y,x) en imagen de entrada)
    imgx = (blockIdx.y * blockDim.x) + threadIdx.x;
    imgy = (blockIdx.x * blockDim.y) + threadIdx.y;

    // Lectura por columna (compartida) / Escritura por fila (global)
    if (imgx < height && imgy < width) {
        d_output[(imgy*height) + imgx] = tile[threadIdx.x][threadIdx.y] ;
    }
}

void transpose_gpu(float * img_in, int width, int height, float * img_out, int algorithm){
    
    switch(algorithm) {
        case 1:
            printf("\n");
            printf("-> Kernel con memoria global\n");
            break;
        case 2:
            printf("\n");
            printf("-> Kernel con memoria compartida\n");
            break;
        case 3:
            printf("\n");    
            printf("-> Kernel con memoria compartida y columna extra\n");
            break;
        default:
            printf("Invocar como: './ej1.x nombre_archivo, algoritmo'\n");
            printf("-> Algoritmo:\n");
            printf("\t 1 - Kernel con memoria global\n");
            printf("\t 2 - Kernel con memoria compartida\n");
            printf("\t 3 - Kernel con memoria compartida y columna extra\n");
            printf("\t 0 - Todos los algoritmos\n");
    }

    // Auxiliar para contar tiempo total
    float t_total = 0;
    
    // Etapa 1: Reserva de Memoria
    CLK_CUEVTS_INIT;
    CLK_CUEVTS_START;
    // Reserva en CPU
    unsigned int size = width * height * sizeof(float);
    float * device_img_in = (float *)malloc(size);
    float * device_img_out = (float *)malloc(size);
    // Reserva en GPU
    CUDA_CHK(cudaMalloc((void**)& device_img_in, size));
    CUDA_CHK(cudaMalloc((void**)& device_img_out, size));
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo transposicion GPU (Reserva de memoria): %f ms\n", t_elap);
    t_total = t_total + t_elap;
    
    // Etapa 2: Transferencia de datos (Host -> Device)
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(device_img_in, img_in, size, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo transposicion GPU (Transferencia de datos (Host -> Device)): %f ms\n", t_elap);
    t_total = t_total + t_elap;

    // Etapa 3: Definir grilla
    int block_amount_x = width / BLOCK_WIDTH + (width % BLOCK_WIDTH != 0); // Division with ceiling
    int block_amount_y = height / BLOCK_HEIGHT + (height % BLOCK_HEIGHT != 0); // Division with ceiling
    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(BLOCK_WIDTH, BLOCK_HEIGHT); // Block dimension

    // Etapa 4 : Lanzar Kernel
    CLK_CUEVTS_START;
    switch(algorithm) {
        case 1:
            transpose_global_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out);
            break;
        case 2:
            transpose_shared_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out);
            break;
        case 3:
            transpose_shared_extra_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out);
            break;
        default:
            transpose_global_kernel<<<tamGrid, tamBlock>>>(device_img_in, width, height, device_img_out);
    }
    // Sincronizar threads antes de parar timers
    cudaDeviceSynchronize(); 
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo transposicion GPU (Kernel): %f ms\n", t_elap);
    t_total = t_total + t_elap;

    // Etapa 5: Transferencia de Datos (Device -> Host)
    CLK_CUEVTS_START;
    CUDA_CHK(cudaMemcpy(img_out, device_img_out, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CLK_CUEVTS_STOP;
    CLK_CUEVTS_ELAPSED;
    printf("Tiempo transposicion GPU (Transferencia de datos (Host <- Device)): %f ms\n", t_elap);
    t_total = t_total + t_elap;
    printf("Tiempo transposicion GPU: %f ms\n", t_total);
    printf("\n");

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_img_in));
    CUDA_CHK(cudaFree(device_img_out));
}
