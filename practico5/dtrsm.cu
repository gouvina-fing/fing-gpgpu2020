#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>    // std::min std::max

using namespace std;

#define TILE_WIDTH   32
#define TILE_HEIGHT  32
// TODO: Definir estos para cargar la shared memory (Cambiarles el nombre?)
#define BLOCK_WIDTH  TILE_WIDTH     // Width of shared memory block
#define BLOCK_HEIGHT TILE_HEIGHT    // Height of shared memory block

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
__global__ void dtrsm_32_shared_kernel() {}

// Ej 2.1 a-2) Kernel para el caso 32 x n con los threads de un warp comunicandose utilizando la primitiva __shfl_sync
__global__ void dtrsm_32_shuffle_kernel() {}

// Ej 2.2) Kernel para el caso 32k x n con los threads de un warp comunicandose a través de la mejor variante de 2.1
// Acá la matriz triangular es de 32k x 32k, y podemos dividirla en k x k tiles de 32 x 32 elementos. Con:
//      - Tiles diagonales (matrices triangulares)
//      - Tiles no diagonales (matrices que no poseen estructura triangular)
// Para resolver n sistemas de 32k:
//      - Cada bloque de threads procesará 32 columnasde B (Recorriendo los tiles de A_{i,j} secuencialmente de izq a der y arriba hacia abajo)
//          Si el tile es diagonal la operacion es idéntica al caso anterior.
//          Si el tile no es diagonal la operación a realizar es la actualización del tile B_{i} mediante una operación DGEMM con tiles de 32x32
//              NOTE: Ver Figura 5. Observar que una operación muy similar es realizada como parte del procedimiento por tiles de la operación DGEMM de la parte anterior.
// NOTE: El parlaelismo grande está en la matriz B (cada fila de bloquecito en B se resuelve en paralelo). Pero las recorridas por los bloques de A son seriales
__global__ void dtrsm_32k_kernel() {}


// Ej 3.3) Kernel que implementa una solución recursiva de DTRSM empleando DGEMM y dividiendo la matriz triangular en tiles de 32x32. 
//         El paso base es DTRSM 32 x n ó DTRSM 32k x n (para un k pequeño) (TODO: Elegir viendo Figura 6 y video. NOTE: No es necesario usar k=32, podemos usar algo más chico)
//         El paso recursivo divide la matriz A en 4 submatrices (Y a B de forma coherente).
// NOTE: Ver letra y Figura 6 para las operaciones con las submatrices
//       Puede ser implementada en CPU (invocando los kernels correspondientes en cada caso, así es moar sencillo)
void dtrsm_recursive() {}

// A y B son arreglos unidimensionales de m × lda y n × ldb elementos.
// Para A el triángulo inferior del bloque superior izquierdo de tamaño m×m debe contener a A en su totalidad (El triangulo superior no es referenciado)
//
// La operación es in-place (los resultados se devuelven en la matriz B)
// TODO: En CuBlas alpha es un double *
void dtrsm_gpu(int algorithm, int m, int n, double alpha, double *A, int lda, double *B, int ldb) {
    // Etapa 1: Reserva de Memoria
    unsigned int size_a = m*m*sizeof(double);
    unsigned int size_b = m*n*sizeof(double);

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
    // TODO: Determinar dimensiones de las grillas
    int block_amount_x = // m / TILE_WIDTH + (m % TILE_WIDTH != 0); // Division with ceiling
    int block_amount_y = // n / TILE_HEIGHT + (n % TILE_HEIGHT != 0); // Division with ceiling
    dim3 tamGrid(block_amount_x, block_amount_y); // Grid dimension
    dim3 tamBlock(TILE_WIDTH, TILE_HEIGHT); // Block dimension

    // Etapa 4 : Lanzar Kernel
    switch(algorithm) {
        case 3: // Versión 32 x n
            dtrsm_32_shared_kernel<<<tamGrid, tamBlock>>>();
            // TODO: cambiar por la más eficiente (shared o shuffle) 
            break;
        case 4: // Versión 32k x n
            dtrsm_32k_kernel<<<tamGrid, tamBlock>>>();
            break;
        case 5: // Versión recursiva.
            dtrsm_recursive();
    }
    cudaDeviceSynchronize();

    // Etapa 5: Transferencia de Datos (Device -> Host)
    CUDA_CHK(cudaMemcpy(B, device_B, size_b, cudaMemcpyDeviceToHost));

    // Etapa 6: Liberación de Memoria
    CUDA_CHK(cudaFree(device_A));
    CUDA_CHK(cudaFree(device_B));
}
