#include "util.h"

// Producto de matrices. Usando doble presición.
// C = βC + αA × B
// Si queremos restarle a C la multiplicación de Axb alpha se define como negativo

// lda, ldb y ldc tienen la cantidad de elementos por fila (width) de cada matriz (lda ≥ k, ldb ≥ n y ldc ≥ n)
// En gral A tiene tantas columnas como lda, B como ldb, etc.
// El sentido de los mismos es si queremos trabajar con submatrices.
// Ejemplo: A 1000x1000, B 1000x1000, C 100x100 C = A'*B' (Con A' y B' las sumatrices de 100x100 de arriba a la izq)
//          m = 100, n = 100, p = 100, lda = 1000, ldb = 1000, ldc = 100
void dgemm_cpu(int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

// NOTE: Para la experimentación se usará lda = k, ldb = n, ldc = n, C = 0, alpha = beta = 1.
// Sin embargo para DTRSM importará una implementación genérica
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

void random_vector(double *A, int n) {
    for (unsigned int i = 0; i < n; ++i) A[i] = (double)rand() / (double)RAND_MAX;
}

void zero_vector(double *A, int n) {
    for (unsigned int i = 0; i < n; ++i) A[i] = 0;
}

// Helper for debugging
void print_matrix_from_vector(double * C, int m, int n) {
    int row;
    for (unsigned int i = 0; i < m; ++i) {
        printf("[");
        row = i*n;
        for (unsigned int j = 0; j < n; ++j)
            printf("%f, ", C[row + j]);
        printf("]\n");
    }
    printf("\n\n");
}

void inicializar_matrices(double **A, double **B, double **C, int m, int p, int n) {
    *A = (double*) malloc(m*p*sizeof(double));
    *B = (double*) malloc(p*n*sizeof(double));
    *C = (double*) malloc(m*n*sizeof(double));

    srand(0); // Inicializa la semilla aleatoria
    random_vector(*A,m*p);
    random_vector(*B,p*n);
    zero_vector(*C,m*n);
}

void liberar_matrices(double **A, double **B, double **C) {
    free(*A); free(*B); free(*C);
}

int main(int argc, char** argv){

	int algorithm, tam1, tam2, tam3;
    double *A, *B, *C;

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
    inicializar_matrices(&A, &B, &C, tam1, tam2, tam3);

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

    liberar_matrices(&A, &B, &C);
    
	return 0;
}

// Notas consulta: TODO: Condensar y borrar:

//En la parte 2 del ejercicio 3 el parlaelismo grande está en la matriz B (cada fila de bloquecito en B se resuelve en paralelo). Pero la recorrida por los bloques de A son seriales

//cublas recibe las matrices en orden Q major (ordenadas por columnas)
//(Porque así viene en Fortran y sus fisicos y metodos numericos), entonces toda la blas está pensada por columnas. Para que el resultado sea el mismo hay que transponer (den la misma salida). El tiempo en teoria sería el mismo. En gral es imposible ganarle a cublas (?)
//hay un parametro que te dice si está transpuesta o no (o si ya le transpusiste)