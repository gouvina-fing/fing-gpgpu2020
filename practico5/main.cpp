#include "util.h"

// Producto de matrices. Usando doble presición.
// C = βC + αA × B
// (Si queremos restarle a C la multiplicación de Axb alpha se define como negativo)
// 
// lda, ldb y ldc tienen la cantidad de elementos por fila (width) de cada matriz (lda ≥ k, ldb ≥ n y ldc ≥ n)
//      En gral A tiene tantas columnas como lda, B como ldb, etc. El sentido de los mismos es si queremos trabajar con submatrices.
//      Ejemplo: A 1000x1000, B 1000x1000, C 100x100 C = A'*B' (Con A' y B' las sumatrices de 100x100 de arriba a la izq)
//          m = 100, n = 100, p = 100, lda = 1000, ldb = 1000, ldc = 100
void dgemm_cpu(int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

// NOTE: Para la experimentación se usará lda = k, ldb = n, ldc = n, C = 0, alpha = beta = 1.
//       Sin embargo para DTRSM importará una implementación genérica
void dgemm_gpu(int algorithm, int m, int n, int p, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

// Resolución de ecuaciones matriciales. Usando doble presición
// A × X = αB, donde α es un escalar, X y B ∈ R^{m×n}, y A ∈ R^{m×m} es una matriz triangular (inferior para esta implementación).
// Esto equivale a resolver n sistemas de ecuaciones de forma Ax_i = b_i, donde b_i es una columna de B y x_i es la solución buscada
// Al ser la matriz triangular el sistema de ecuaciones lineales ya viene "escalerizado".
// 
// A y B son arreglos unidimensionales de m × lda y n × ldb elementos respectivamente.
// Para A el triángulo inferior del bloque superior izquierdo de tamaño m×m debe contener a A en su totalidad (El triangulo superior no es referenciado)
//
// La operación es in-place (los resultados se devuelven en la matriz B)
void dtrsm_gpu(int algorithm, int m, int n, double alpha, double *A, int lda, double *B, int ldb);

void dtrsm_cublas(int m, int n, const double *alpha, double *A, int lda, double *B, int ldb);

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

// TODO: Algo de esto para matrices triangulares con determinante no nulo :'v
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

void transponer_vector(double *A, double *transposeA, int m, int n) {
    int row;
    for (unsigned int i = 0; i < m; ++i) {
        row = i*n;
        for (unsigned int j = 0; j < n; ++j)
            transposeA[j*m + i] = A[row + j];
    }
}

int main(int argc, char** argv){

	int algorithm, tam1, tam2, tam3;
    double *A, *B, *C, *transposeA, *transposeB;
    const double alpha = 1.0;

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
    double testA[3*3] = { 1.0, 0, 0, 2.0, 3.0, 0, 4.0, 5.0, 6.0 };
    double testB[4*3] = { 1, 10, 50, 100, 8, 80, 400, 800, 32, 320, 1600, 3200 };
    switch(algorithm) {
        case 1: // DGEMM con memoria global
        case 2: // DGEMM con memoria comaprtida
            dgemm_gpu(algorithm, tam1, tam3, tam2, alpha, A, tam2, B, tam3, 1, C, tam3);
            break;
        case 3: // DTRSM A (32 x 32) B (32 x tam1)
            dtrsm_gpu(algorithm, 32, tam1, alpha, A, 32, B, 32);
            break;
        case 4: // DTRSM con bloques de k*32 recorriendo secuencialmente A (tam1 x tam1) B (tam1 x tam2)
        case 5: // DTRSM versión recursiva A (tam1 x tam1) B (tam1 x tam2)
            dtrsm_gpu(algorithm, tam1, tam2, alpha, A, tam1, B, tam1);
            break;
        case 6: // DTRSM de la biblioteca CuBlas A (tam1 x tam1) B (tam1 x tam2)
            // dtrsm_cublas(tam1, tam2, &alpha, A, tam1, B, tam1);

            tam1 = 3;
            tam2 = 4;

            transposeA = (double*) malloc(tam1*tam1*sizeof(double));
            transposeB = (double*) malloc(tam2*tam2*sizeof(double));

            printf("A:\n");
            print_matrix_from_vector(testA,tam1,tam1);
            //double testA[3*3] = { 1.0, 2.0, 4.0, 0, 3.0, 5.0, 0, 0, 6.0 };

            transponer_vector(testA, transposeA, tam1,tam1);
            printf("A Transpose:\n");
            print_matrix_from_vector(transposeA,tam1,tam1);

            printf("\n");

            printf("b:\n");
            print_matrix_from_vector(testB,tam1,tam2);
            //double testB[4*3] = { 1, 8, 32, 10, 80, 320, 50, 400, 1600, 100, 800, 3200 };

            printf("\n");

            printf("Result:\n");

            transponer_vector(testB, transposeB, tam1,tam2);
            printf("B Transpose:\n");
            print_matrix_from_vector(transposeB,tam2,tam1);

            dtrsm_cublas(tam1, tam2, &alpha, transposeA, tam1, transposeB, tam1);

            transponer_vector(transposeB, testB, tam2, tam1);

            print_matrix_from_vector(testB,tam1,tam2);

            // TODO: Invocar a CuBlas
            // Notas consulta:
                // CuBlas recibe las matrices en orden Q major (ordenadas por columnas) (Porque así viene en Fortran y sus fisicos y metodos numericos)
                // Entonces toda la blas está pensada por columnas. Para que el resultado sea el mismo hay que transponer (den la misma salida). El tiempo en teoria sería el mismo.
                // Hay un parametro que te dice si está transpuesta o no (o si ya le transpusiste)
                // En general es imposible superar a CuBlas en tiempos (?)
                // Transponer en CPU antes de llamar la cosa
            break;
        case 0: // DGEMM CPU
            // NOTE: No hacer un "Todos" porque todo acá sobreescribe en los datos de lectura
            dgemm_cpu(tam1, tam3, tam2, alpha, A, tam2, B, tam3, 1, C, tam3);
            break;
        default:
            break;
    }

    
    

    liberar_matrices(&A, &B, &C);
    
	return 0;
}