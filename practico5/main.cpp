#include "util.h"

// Producto de matrices. Usando doble presición.
// C = βC + αA × B
// Si queremos restarle a C la multiplicación de Axb alpha se define como negativo

// 1.b
// Cada bloque calcula un tile de C, cada hilo un elemento de C, van pasando tiles de A y B a memoria compartida, multiplicando y cargando otro.
// Cada hilo guarda su resultado de C en un registro
// Asumimos que los tamaños del tile siempre son multiplos del tamaño de bloque
// https://spatial-lang.org/gemm
void dgemm_gpu(int algorithm, int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

// Resolución de un conjunto de sistemas de ecuaciones lineales triangulares (con un sitema ya "escalerizado"). Usando doble presición.
void dtrsm_gpu(int algorithm, int m, int n, double *alpha, double *A, int lda, double *B, int ldb);

int print_trace_format() {
    printf("Invocar como: './labgpu20.x algoritmo [tam1] [tam2] [tam3]'\n");
    printf("-> Algoritmo:\n");
        printf("\t 1 - DGEMM en memoria global A (tam1 x tam2) B (tam2 x tam3)\n");
        printf("\t 2 - DGEMM con memoria compartida A (tam1 x tam2) B (tam2 x tam3)\n");
        printf("\t 3 - DTRSM A (32 x 32) B (32 x tam1)\n");
        printf("\t 4 - DTRSM con bloques de k*32 recorriendo secuencialmente A (tam1 x tam1) B (tam1 x tam2)\n");
        printf("\t 5 - DTRSM versión recursiva A (tam1 x tam1) B (tam1 x tam2)\n");
        printf("\t 6 - DTRSM de la biblioteca CuBlas A (tam1 x tam1) B (tam1 x tam2)\n");
        printf("\t 0 - Todos los algoritmos\n");
    return 1;
}

// lda, ldb y ldc dicen cuantos elementos hay en cada fila de los arreglos bidimensionales que contengan estas matrices
// En gral A tiene tantas columnas como lda, B como ldb, etc.
// El sentido de los mismos es si queremos trabajar con submatrices.
// lda ≥ k, ldb ≥ n y ldc ≥ n
void dgemm_cpu(int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc) {
    int i,j,k;
    double alpha_a;

    for(i = 0; i < m; ++i) {
        for(j = 0; j < n; ++j)
            C[i*ldc + j] *= beta;

        for(k = 0; k < p; ++k) {
            alpha_a = alpha*A[i*lda + k];
            for(j = 0; j < n; ++j)
                C[i*ldc + j] += alpha_a*B[k*ldb + j];
        }
    }
}

int main(int argc, char** argv){

	int algorithm, tam1, tam2, tam3;
    double *A, *B; // TODO: Como inicializamos esta verga?
    double *C; // TODO: Imagino que inicializar con 0

    // Do stuff that may throw or fail
    
    // Get algorithm
    if (argc < 3) return print_trace_format();
    algorithm = atoi(argv[1]);

    // Get the rest of parameters
    tam1 = atoi(argv[2]); // tam1 is always passed
	if(algorithm != 3) {
        if (argc < 4) return print_trace_format();
        tam2 = atoi(argv[3]); // tam2 is not required for algorithm 3

        if((algorithm == 0) || (algorithm == 1) || (algorithm == 2)) {
            if (argc < 5) return print_trace_format();
            tam3 = atoi(argv[4]); // tam3 is not required for algorithms 3, 4, 5 and 6
        }
    }

    // DGEMM (Algorithm 1 | 2):
    // m = tam1
    // k | p = tam2
    // n = tam3

    // Execute algorithm
    switch(algorithm) {
        case 1:
        case 2:
            dgemm_gpu(algorithm, tam1, tam3, tam2, 1.0, A, tam2, B, tam3, 1, C, tam3);
            break;
        case 3:
            break;
        case 4:
            break;
        case 5:
            break;
        case 6:
            break;
        case 0:
            dgemm_cpu(tam1, tam3, tam2, 1.0, A, tam2, B, tam3, 1, C, tam3);
            break;
        default:
            break;
    }
    
	return 0;
}

