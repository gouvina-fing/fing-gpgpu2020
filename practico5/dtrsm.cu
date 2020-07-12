#include "util.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <algorithm>    // std::min std::max

using namespace std;

#define TILE_WIDTH   32
#define TILE_HEIGHT  32

// Resolución de ecuaciones matriciales. Usando doble presición
// A × X = αB, donde α es un escalar, X y B ∈ R^{m×n}, y A ∈ R^{m×m} es una matriz triangular (inferior para esta implementación).
// Esto equivale a resolver n sistemas de ecuaciones de forma Ax_i = b_i, donde b_i es una columna de B y x_i es la solución buscada
// Al ser la matriz triangular el sistema de ecuaciones lineales ya viene "escalerizado".

// Ej 2.1 a) Caso 32 x n
// Para resolver estos sistemas:
//      - Cada bloque de threads debe sobreescribir un tile de B con el resultado de la operación.
//      - Cada warp del bloque procesa una columna de 32 elementos (resuelve uno de los n sistemas de ecuaciones). Como todos usan A hay que guardarla en memoria rápida.
//      - Cada thread del warp calcula un elemento de la columna().
//      - Cada thread lee datos calculados por los threads del warp del índice anterior. Para compartir datos entre los hilos del warp tenemos las siguientes opciones:

// Ej 2.1 a-1) Kernel para el caso 32 x n con los threads de un warp comunicandose a través memoria compartida
// El paralelismo a nivel de warps es implicito, porque dentro de un warp se avanza en el código secuencialmente
__global__ void dtrsm_32_shared_kernel(const double alpha, double *d_A, int lda, double *d_B, int ldb, int stride_A, int stride_B) {
    __shared__ double shared_A[TILE_WIDTH][TILE_HEIGHT];
    __shared__ double tile_B[TILE_WIDTH][TILE_HEIGHT];

    double aux;
    int x, y, row_b, memory_index_x, memory_index_y;

    x = (blockIdx.x * blockDim.x) + threadIdx.x; // Column
    y = (blockIdx.y * blockDim.y) + threadIdx.y; // Row
    memory_index_x = threadIdx.x;
    memory_index_y = threadIdx.y;
    row_b = x*ldb;

    // Cada bloque guarda su pixel de A en memoria compartida
    shared_A[memory_index_x][memory_index_y] = d_A[memory_index_x*lda + memory_index_y + stride_A];
    aux = alpha*d_B[row_b + y + stride_B];
    __syncthreads();

    for(int k = 0; k <= TILE_WIDTH; ++k) {
        if(k == memory_index_x) {
            // Se llegó a la diagonal de A, la incógnita queda resuelta y se guarda su resultado
            tile_B[k][memory_index_y] = aux/shared_A[k][k];
        }
        __syncwarp();
        if(k < memory_index_x) {
            // Se va acumulando la resta de productos mientras se sube por la diagonal de A.
            aux -= shared_A[memory_index_x][k]*tile_B[k][memory_index_y];
        }
    }
    d_B[row_b + y + stride_B] = tile_B[memory_index_x][memory_index_y];
}

// Ej 2.1 a-2) Kernel para el caso 32 x n con los threads de un warp comunicandose utilizando la primitiva __shfl_sync
__global__ void dtrsm_32_shuffle_kernel(const double alpha, double *d_A, int lda, double *d_B, int ldb, int stride_A, int stride_B) {
    __shared__ double shared_A[TILE_WIDTH][TILE_HEIGHT];

    double result, aux, aux2;
    int x, y, row_b, memory_index_x, memory_index_y;

    x = (blockIdx.x * blockDim.x) + threadIdx.x; // Row
    y = (blockIdx.y * blockDim.y) + threadIdx.y; // Column
    memory_index_x = threadIdx.x;
    memory_index_y = threadIdx.y;
    row_b = x*ldb;

    // Cada bloque guarda su pixel de A en memoria compartida
    shared_A[memory_index_x][memory_index_y] = d_A[memory_index_x*lda + memory_index_y + stride_A];
    aux = alpha*d_B[row_b + y + stride_B];

    __syncthreads();

    result = 0;

    for(int k = 0; k <= TILE_WIDTH; ++k) {
        if(k == memory_index_x) {
            // Se llegó a la diagonal de A, la incógnita queda resuelta y se guarda su resultado
            result = aux/shared_A[k][k];
        }
        __syncwarp();
        aux2 = __shfl_sync(0xffffffff, result, k);
        if(k < memory_index_x) {
            // Se va acumulando la resta de productos mientras se sube por la diagonal de A.
            aux -= shared_A[memory_index_x][k]*aux2;
        }
    }
    d_B[row_b + y + stride_B] = result;
}

__global__ void dgemm_shared_kernel(int p, const double alpha, double *d_A, int lda, double *d_B, int ldb, double beta, double *d_C, int ldc, int stride_A, int stride_B, int stride_C);

// Ej 2.2) Función para el caso 32k x n con los threads de un warp comunicandose a través de la mejor variante de 2.1
// Acá la matriz triangular es de 32k x 32k, y podemos dividirla en k x k tiles de 32 x 32 elementos. Con:
//      - Tiles diagonales (matrices triangulares)
//      - Tiles no diagonales (matrices que no poseen estructura triangular)
// Para resolver n sistemas de 32k:
//      - Cada bloque de threads procesará 32 columnasde B (Recorriendo los tiles de A_{i,j} secuencialmente de izq a der y arriba hacia abajo)
//          Si el tile es diagonal la operacion es idéntica al caso anterior.
//          Si el tile no es diagonal la operación a realizar es la actualización del tile B_{i} mediante una operación DGEMM con tiles de 32x32
void dtrsm_32k(int block_amount_x, int block_amount_y, const double alpha, double *d_A, int lda, double *d_B, int ldb, int meta_stride_A, int meta_stride_B) {
    // A es de 32k x 32k. En donde k == block_amount_x
    // B es de 32k x n. En donde k == block_amount_x y n = 32*block_amount_y

    int stride_A, stride_B, stride_C;
    dim3 tamGrid(1, block_amount_y); // Grid dimension
    dim3 tamGridDGEMM(block_amount_y, 1); // Grid dimension para DGEMM, accede por filas en lugar de columnas por lo que está espejado
    dim3 tamBlock(TILE_WIDTH, TILE_HEIGHT); // Block dimension

    for(int i = 0; i < block_amount_x; ++i) {
        stride_A = meta_stride_A + 32*i*lda; // Move the stride in A to the next block of rows.
        stride_B = meta_stride_B; // Move the stride in B to the previous block of rows (Not used when i = 0).
        stride_C = meta_stride_B + 32*i*ldb; // Move the stride in C to the next block of rows.
        for(int j = 0; j <= i; ++j) {
            if (i == j) { // Diagonal
                dtrsm_32_shared_kernel<<<tamGrid, tamBlock>>>(alpha, d_A, lda, d_B, ldb, stride_A, stride_C);
            } else { // No diagonal
                // Bi = Bi - Aij * Bj
                // Bi = 32 x n (fila superior). Bj = 32 x n (fila actual a actualizar). A = 32 x 32. p == n
                dgemm_shared_kernel<<<tamGridDGEMM, tamBlock>>>(32, -1.0, d_A, lda, d_B, ldb, 1.0, d_B, ldb, stride_A, stride_B, stride_C);
            }
            stride_A += 32; // Move the stride in A to the next column block
            stride_B += 32*ldb;
        }
    }
}


// Ej 3.3) Kernel que implementa una solución recursiva de DTRSM empleando DGEMM y dividiendo la matriz triangular en tiles de 32x32. 
//         El paso base es DTRSM 32 x n ó DTRSM 32k x n (para un k pequeño)
//         El paso recursivo divide la matriz A en 4 submatrices (Y a B de forma coherente).
void dtrsm_recursive(int m, int block_amount_y, const double alpha, double *d_A, int lda, double *d_B, int ldb, int stride_A, int stride_B) {
    if(m == 64) { // Paso base, A 32*2 x 32*2
        dtrsm_32k(2, block_amount_y, alpha, d_A, lda, d_B, ldb, stride_A, stride_B);
    } else { // Paso recursivo
        // A y B se parten en: |A11  0 |  |B1|
        //                     |A21 A22|  |B2|

        m /= 2;
        dim3 tamGridDGEMM(block_amount_y, m/32); // Grid dimension para DGEMM, accede por filas en lugar de columnas por lo que está espejado
        dim3 tamBlock(TILE_WIDTH, TILE_HEIGHT); // Block dimension
        
        // Se procesa A11, manteniendo direcciones de memoria.
        dtrsm_recursive(m, block_amount_y, alpha, d_A, lda, d_B, ldb, stride_A, stride_B);

        // Se procesa A21 (DGEMM), shifteando las direcciones de memoria al bloque de filas de abajo.
        dgemm_shared_kernel<<<tamGridDGEMM, tamBlock>>>(m, -1.0, d_A, lda, d_B, ldb, 1.0, d_B, ldb, stride_A + m*lda, stride_B, stride_B + m*ldb);

        // Se procesa A22, shifteando las direcciones de memoria al bloque de filas de abajo y A m columnas hacia la derecha.
        dtrsm_recursive(m, block_amount_y, alpha, d_A, lda, d_B, ldb, stride_A + m*lda + m, stride_B + m*ldb);
    }
}

// A y B son arreglos unidimensionales de m × lda y n × ldb elementos.
// Para A el triángulo inferior del bloque superior izquierdo de tamaño m×m debe contener a A en su totalidad (El triangulo superior no es referenciado)
// La operación es in-place (los resultados se devuelven en la matriz B)
void dtrsm_gpu(int algorithm, int m, int n, const double alpha, double *A, int lda, double *B, int ldb) {
    // Etapa 1: Reserva de Memoria
    unsigned int size_a = m*lda*sizeof(double);
    unsigned int size_b = m*ldb*sizeof(double);

    // Reserva en CPU
    double * device_A = (double *)malloc(size_a);
    double * device_B = (double *)malloc(size_b);
    
    // Reserva en GPU
    CUDA_CHK(cudaMalloc((void**)& device_A, size_a));
    CUDA_CHK(cudaMalloc((void**)& device_B, size_b));

    // Etapa 2: Transferencia de datos (Host -> Device)
    CUDA_CHK(cudaMemcpy(device_A, A, size_a, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia
    CUDA_CHK(cudaMemcpy(device_B, B, size_b, cudaMemcpyHostToDevice));

    // Etapa 3: Definir grilla
    // Se crea una grilla con las dimensiones de B
    int block_amount_x = m / TILE_WIDTH + (m % TILE_WIDTH != 0); // Division with ceiling
    int block_amount_y = n / TILE_HEIGHT + (n % TILE_HEIGHT != 0); // Division with ceiling
    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(TILE_WIDTH, TILE_HEIGHT); // Block dimension

    // Etapa 4 : Lanzar Kernel
    switch(algorithm) {
        case 3: // Versión 32 x n
            dtrsm_32_shared_kernel<<<tamGrid, tamBlock>>>(alpha, device_A, lda, device_B, ldb, 0, 0);
            break;
        case 4: // Versión 32k x n
            dtrsm_32k(block_amount_x, block_amount_y, alpha, device_A, lda, device_B, ldb, 0, 0);
            break;
        case 5: // Versión recursiva.
            dtrsm_recursive(m, block_amount_y, alpha, device_A, lda, device_B, ldb, 0, 0);
            break;
        case 7: // Versión 32 x n Shuffle/Shared (la menos eficiente)
            dtrsm_32_shuffle_kernel<<<tamGrid, tamBlock>>>(alpha, device_A, lda, device_B, ldb, 0, 0);
    }
    cudaDeviceSynchronize();

    // Etapa 5: Transferencia de Datos (Device -> Host)
    CUDA_CHK(cudaMemcpy(B, device_B, size_b, cudaMemcpyDeviceToHost));

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_A));
    CUDA_CHK(cudaFree(device_B));
}

void dtrsm_cublas(int m, int n, const double *alpha, double *A, int lda, double *B, int ldb) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t handle;

    // Etapa 1: Reserva de Memoria
    unsigned int size_a = m*lda*sizeof(double);
    unsigned int size_b = ldb*n*sizeof(double);

    // Reserva en CPU
    double * device_A = (double *)malloc(size_a);
    double * device_B = (double *)malloc(size_b);

    // Reserva en GPU
    CUDA_CHK(cudaMalloc((void**)& device_A, size_a));
    CUDA_CHK(cudaMalloc((void**)& device_B, size_b));

    // Etapa 2: Crear Handle de CuBlas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    // Etapa 3: Transferencia de datos (Host -> Device)
    status = cublasSetMatrix(m, m, sizeof(double), A, lda, device_A, lda);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data download A failed\n");
        CUDA_CHK(cudaFree(device_A));
        cublasDestroy(handle);
        return;
    }
    status = cublasSetMatrix (m, n, sizeof(double), B, ldb, device_B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data download B failed\n");
        CUDA_CHK(cudaFree(device_A));
        CUDA_CHK(cudaFree(device_B));
        cublasDestroy(handle);
        return;
    }

    // Etapa 4 : Lanzar Kernel
    status = cublasDtrsm(
        handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        m, n, alpha, device_A, lda, device_B, ldb
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("DTRSM operation failed\n");
        CUDA_CHK(cudaFree(device_A));
        CUDA_CHK(cudaFree(device_B));
        cublasDestroy(handle);
        return;
    }

    // Etapa 5: Transferencia de Datos (Device -> Host)
    status = cublasGetMatrix (m, n, sizeof(double), device_B, ldb, B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        cublasDestroy(handle);
    }

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_A));
    CUDA_CHK(cudaFree(device_B));
}