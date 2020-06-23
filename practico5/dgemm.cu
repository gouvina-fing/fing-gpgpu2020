#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

#define TILE_WIDTH   32
#define TILE_HEIGHT  32

// Ej 1a) Kernel
__global__ void dgemm_global_kernel() {}

// Ej 1b) Kernel
__global__ void dgemm_shared_kernel() {}

void dgemm_gpu(int algorithm, int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
    // Etapa 1: Reserva de Memoria

    // Etapa 2: Transferencia de datos (Host -> Device)

    // Etapa 3: Definir grilla

    // Etapa 4 : Lanzar Kernel
    // CLK_CUEVTS_START;
    switch(algorithm) {
        case 1:
            dgemm_global_kernel<<<tamGrid, tamBlock>>>();
            break;
        case 2:
            dgemm_shared_kernel<<<tamGrid, tamBlock>>>();
            break;
    }
    // Sincronizar threads antes de parar timers
    cudaDeviceSynchronize();
    // CLK_CUEVTS_STOP;
    // CLK_CUEVTS_ELAPSED;
    // printf("Tiempo transposicion GPU (Kernel): %f ms\n", t_elap);
    // t_total = t_total + t_elap;

    // Etapa 5: Transferencia de Datos (Device -> Host)

    // Etapa 6: Liberación de Memoria
}
