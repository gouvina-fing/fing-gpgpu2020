#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

#define TILE_WIDTH   32
#define TILE_HEIGHT  32

// Producto de matrices. Usando doble presición.
// C = βC + αA × B
// Si queremos restarle a C la multiplicación de Axb alpha se define como negativo

// lda, ldb y ldc tienen la cantidad de elementos por fila (width) de cada matriz (lda ≥ k, ldb ≥ n y ldc ≥ n)
// En gral A tiene tantas columnas como lda, B como ldb, etc.
// El sentido de los mismos es si queremos trabajar con submatrices.
// Ejemplo: A 1000x1000, B 1000x1000, C 100x100 C = A'*B' (Con A' y B' las sumatrices de 100x100 de arriba a la izq)
//          m = 100, n = 100, p = 100, lda = 1000, ldb = 1000, ldc = 100

// Link relevante: https://spatial-lang.org/gemm

// Ej 1a)
// Cada bloque calcula un tile de C, cada hilo un elemento de C.
// No emplea memoria compartida ni otras optimizaciones.
// Asumimos que los tamaños del tile siempre son multiplos del tamaño de bloque
__global__ void dgemm_global_kernel(int p, const double alpha, double *d_A, int lda, double *d_B, int ldb, double beta, double *d_C, int ldc) {
    int x, y, k, row_a, row_b, row_c;
    double alpha_a, result;

    x = (blockIdx.x * blockDim.x) + threadIdx.x; // Column
    y = (blockIdx.y * blockDim.y) + threadIdx.y; // Row
    row_a = y*lda;
    row_c = y*ldc;
    result = d_C[row_c + x]*beta;

    for(k = 0; k < p; ++k) {
        row_b = k*ldb;
        alpha_a = alpha*d_A[row_a + k];

        result += alpha_a*d_B[row_b + x];
    }
    d_C[row_c + x] = result;
}

// Ej 1b)
// Cada bloque calcula un tile de C, cada hilo un elemento de C.
// Cada bloque va pasando tiles de A y B a memoria compartida, multiplicandolos, acumulando el resultado en un registro y luego cargando otros tiles de A y B.
// Asumimos que los tamaños del tile siempre son multiplos del tamaño de bloque
__global__ void dgemm_shared_kernel(int p, const double alpha, double *d_A, int lda, double *d_B, int ldb, double beta, double *d_C, int ldc) {
    __shared__ double tile_A[TILE_WIDTH][TILE_HEIGHT];
    __shared__ double tile_B[TILE_WIDTH][TILE_HEIGHT];

    int x, y, k, row_a, row_c, memory_index_x, memory_index_y, idx, idy;
    double alpha_a, result;

    x = (blockIdx.x * blockDim.x) + threadIdx.x; // Column
    y = (blockIdx.y * blockDim.y) + threadIdx.y; // Row
    row_a = y*lda;
    row_c = y*ldc;
    result = d_C[row_c + x]*beta;

    memory_index_x = threadIdx.x;
    memory_index_y = threadIdx.y;

    // Iteramos por cada bloque en las filas de A y columnas de B
    for(int step = 0; step < p; step+=32) {
        idx = step + memory_index_x;
        idy = step + memory_index_y;

        // Los hilos guardan el bloque en memoria compartida
        tile_A[memory_index_y][memory_index_x] = d_A[row_a + idx];
        tile_B[memory_index_y][memory_index_x] = d_B[idy*ldb + x];
        __syncthreads();

        // Se opera acediendo a los bloques previamente guardados
        for(k = 0; k < 32; ++k) {
            alpha_a = alpha*tile_A[memory_index_y][k];
            result += alpha_a*tile_B[k][memory_index_x];
        }
        // Se sincroniza para evitar que la memoria compartida sea editada mientras aún se usa para operar
        __syncthreads();
    }

    d_C[row_c + x] = result;
}

void dgemm_gpu(int algorithm, int m, int n, int p, const double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
    // Etapa 1: Reserva de Memoria
    unsigned int size_a = m*p*sizeof(double);
    unsigned int size_b = p*n*sizeof(double);
    unsigned int size_c = m*n*sizeof(double);

    // Reserva en CPU
    double * device_A = (double *)malloc(size_a);
    double * device_B = (double *)malloc(size_b);
    double * device_C = (double *)malloc(size_c);
    
    // Reserva en GPU
    CUDA_CHK(cudaMalloc((void**)& device_A, size_a));
    CUDA_CHK(cudaMalloc((void**)& device_B, size_b));
    CUDA_CHK(cudaMalloc((void**)& device_C, size_c));

    // Etapa 2: Transferencia de datos (Host -> Device)
    CUDA_CHK(cudaMemcpy(device_A, A, size_a, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CUDA_CHK(cudaMemcpy(device_B, B, size_b, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(device_C, C, size_c, cudaMemcpyHostToDevice));

    // Etapa 3: Definir grilla
    // Se crea una grilla con las dimensiones de C (un hilo por valor en C)
    int block_amount_x = m / TILE_WIDTH + (m % TILE_WIDTH != 0); // Division with ceiling
    int block_amount_y = n / TILE_HEIGHT + (n % TILE_HEIGHT != 0); // Division with ceiling
    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(TILE_WIDTH, TILE_HEIGHT); // Block dimension

    // Etapa 4 : Lanzar Kernel
    switch(algorithm) {
        case 1:
            dgemm_global_kernel<<<tamGrid, tamBlock>>>(p, alpha, device_A, lda, device_B, ldb, beta, device_C, ldc);
            break;
        case 2:
            dgemm_shared_kernel<<<tamGrid, tamBlock>>>(p, alpha, device_A, lda, device_B, ldb, beta, device_C, ldc);
    }
    cudaDeviceSynchronize();

    // Etapa 5: Transferencia de Datos (Device -> Host)
    CUDA_CHK(cudaMemcpy(C, device_C, size_c, cudaMemcpyDeviceToHost));

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_A));
    CUDA_CHK(cudaFree(device_B));
    CUDA_CHK(cudaFree(device_C));
}

void dgemm_cpu(int m, int n, int p, const double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
    int i, j, k, row_a, row_b, row_c;
    double alpha_a;

    for(i = 0; i < m; ++i) {
        row_a = i*lda;
        row_c = i*ldc;
        for(j = 0; j < n; ++j)
            C[row_c + j] *= beta;

        for(k = 0; k < p; ++k) {
            row_b = k*ldb;
            alpha_a = alpha*A[row_a + k];
            for(j = 0; j < n; ++j)
                C[row_c + j] += alpha_a*B[row_b + j];
        }
    }
}