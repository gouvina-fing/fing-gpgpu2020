#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define TIME(t_i,t_f) ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0) - \
                      ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0);
#define RUNS 10

void random_matriz(float **A, int m, int n) {
    for (unsigned int i = 0; i < m; i++) 
        for (unsigned int j = 0; j < n; j++) 
                A[i][j] = (float)rand() / (float)RAND_MAX;
}

/* Ej A)
*   Construir la multiplicacion de matrices con el patron de acceso usual (computando C_{ij} antes de avanzar a la siguiente).
*   Las matrices no tienen porque ser cuadradas.
*   C = A x B. C_{ij} = Sum{k=1}{p}(A_{ik} x B_{kj})
*/
void mult_simple(float ** A, float ** B, float ** C, int m, int p, int n) {
    // Initializing elements of matrix C to 0.
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
            C[i][j] = 0;

    // Multiplying A and B and storing in C.
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            for (unsigned int k = 0; k < p; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Ej B)
*   Construir la multiplicacion de matrices que acceda por fila tanto a la matriz A como a la matriz B (en lugar de por columna a B)
*   Las matrices no tienen porque ser cuadradas.
*   C = A x B. C_{ij} = Sum{k=1}{p}(A_{ik} x B_{kj})
*/
void mult_por_filas(float ** A, float ** B, float ** C, int m, int p, int n) {
    // Initializing elements of matrix mult to 0.
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
            C[i][j] = 0;

    // Multiplying A and B and storing in C.
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int k = 0; k < p; ++k) {
            for (unsigned int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Ej C)
*   Construir la multiplicacion de matrices "por bloques" de tamanio variable (nb = 64, 128, 256, 512)
*   Las matrices no tienen porque ser cuadradas.
*   C = A x B. C_{ij} = Sum{k=1}{p}(A_{ik} x B_{kj})
*/
void mult_por_bloques(float ** A, float ** B, float ** C, int m, int p, int n, int nb) {
    int blockSize = nb;

    // Initializing elements of matrix mult to 0.
    for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < n; ++j)
            C[i][j] = 0;

    for (unsigned int bi=0; bi<m; bi+=blockSize)
        for (unsigned int bj=0; bj<n; bj+=blockSize)
            for (unsigned int bk=0; bk<p; bk+=blockSize)
                for (unsigned int i=0; i<blockSize; i++)
                    for (unsigned int k=0; k<blockSize; k++)
                        for (unsigned int j=0; j<blockSize; j++)
                            C[bi+i][bj+j] += A[bi+i][bk+k]*B[bk+k][bj+j];
}

// Helper for debugging
void print_matrix(float ** C, int m, int n) {
    for (unsigned int i = 0; i < m; ++i) {
        printf("[");
        for (unsigned int j = 0; j < n; ++j)
            printf("%f, ", C[i][j]);
        printf("]\n");
    }
    printf("\n\n");
}

void inicializar_matrices(float ***A, float ***B, float ***C, int m, int p, int n) {
    *A = (float**) malloc(m*sizeof(float*));
    for (int i = 0; i < m; ++i) (*A)[i] = (float*) malloc(p*sizeof(float)); 

    *B = (float**) malloc(p*sizeof(float*));
    for (int i = 0; i < p; ++i) (*B)[i] = (float*) malloc(n*sizeof(float)); 

    *C = (float**) malloc(m*sizeof(float*));
    for (int i = 0; i < m; ++i) (*C)[i] = (float*) malloc(n*sizeof(float)); 

    srand(0); // Inicializa la semilla aleatoria
    random_matriz(*A,m,p);
    random_matriz(*B,p,n);
}

void liberar_matrices(float ***A, float ***B, float ***C, int m, int p, int n) {
    for (int i = 0; i < m; ++i) free((*A)[i]);
    for (int i = 0; i < p; ++i) free((*B)[i]);
    for (int i = 0; i < n; ++i) free((*C)[i]);
    free(*A);
    free(*B);
    free(*C);
}

double corrida_simple(float ** A, float ** B, float ** C, int m, int p, int n) {
    struct timeval t_i, t_f;

    gettimeofday(&t_i, NULL);
    mult_simple(A,B,C,m,p,n);
    gettimeofday(&t_f, NULL);

    return TIME(t_i,t_f);
}

double corrida_por_filas(float ** A, float ** B, float ** C, int m, int p, int n) {
    struct timeval t_i, t_f;

    gettimeofday(&t_i, NULL);
    mult_por_filas(A,B,C,m,p,n);
    gettimeofday(&t_f, NULL);

    return TIME(t_i,t_f);
}

double corrida_por_bloques(float ** A, float ** B, float ** C, int m, int p, int n, int nb) {
    struct timeval t_i, t_f;

    gettimeofday(&t_i, NULL);
    mult_por_bloques(A,B,C,m,p,n,nb);
    gettimeofday(&t_f, NULL);

    return TIME(t_i,t_f);
}

void benchmark_simple(float ** A, float ** B, float ** C, int m, int p, int n) {
    inicializar_matrices(&A, &B, &C, m, p, n);
    printf("0,%i,%i,%i,0", m, p, n);
    for (unsigned int j = 0; j < RUNS; j++) {
        printf(",%f", corrida_simple(A,B,C,m,p,n));
    }
    printf("\n");
    liberar_matrices(&A, &B, &C, m, p, n);
}

void benchmark_por_filas(float ** A, float ** B, float ** C, int m, int p, int n) {
    inicializar_matrices(&A, &B, &C, m, p, n);
    printf("1,%i,%i,%i,0", m, p, n);
    for (unsigned int j = 0; j < RUNS; j++) {
        printf(",%f", corrida_por_filas(A,B,C,m,p,n));
    }
    printf("\n");
    liberar_matrices(&A, &B, &C, m, p, n);
}

void benchmark_por_bloques(float ** A, float ** B, float ** C, int m, int p, int n, int nb) {
    inicializar_matrices(&A, &B, &C, m, p, n);
    printf("2,%i,%i,%i,%i", m, p, n, nb);
    for (unsigned int j = 0; j < RUNS; j++) {
        printf(",%f", corrida_por_bloques(A,B,C,m,p,n,nb));
    }
    printf("\n");
    liberar_matrices(&A, &B, &C, m, p, n);
}

int main(int argc, char *argv[]){
    
    int n = 512, m = 512, p = 512, nb=16;
    float **A, **B, **C; 

    // Se desea correr un ejemplo particular
    if (argc > 1) {
        if (argc < 5) {
            printf("Uso: ./ej3 n m p nb \n");
            exit(1);
        }
        if(argc == 5){
            m  = atoi(argv[1]);
            n  = atoi(argv[2]);
            p  = atoi(argv[3]);
            nb = atoi(argv[4]);
        }

        inicializar_matrices(&A, &B, &C, m, p, n);

        // Evaluar tiempo de mult_simpĺe
        double t_gemm_simple = corrida_simple(A,B,C,m,p,n);

        // Evaluar tiempo de mult_por_filas
        double t_gemm_fil = corrida_por_filas(A,B,C,m,p,n);

        // Evaluar tiempo de mult_por_bloques
        double t_gemm_bloq = corrida_por_bloques(A,B,C,m,p,n,nb);

        printf("Tamano: (%i,%i,%i,%i) Tiempo simple:    %f ms\n", m, n, p, nb, t_gemm_simple);
        printf("Tamano: (%i,%i,%i,%i) Tiempo filas:    %f ms\n", m, n, p, nb, t_gemm_fil);
        printf("Tamano: (%i,%i,%i,%i) Tiempo bloques: %f ms\n", m, n, p, nb, t_gemm_bloq);

        liberar_matrices(&A, &B, &C, m, p, n);

    } else { // Se desea correr el benchmark para todos los ejemplos
        
        // Evaluar tiempo de mult_simpĺe
        m = 2048; p = 2048; n = 2048;
        benchmark_simple(A,B,C,m,p,n);

        m = 2048; p = 1024; n = 2048;
        benchmark_simple(A,B,C,m,p,n);

        m = 1024; p = 2048; n = 1024;
        benchmark_simple(A,B,C,m,p,n);

        // Evaluar tiempo de mult_por_filas
        m = 2048; p = 2048; n = 2048;
        benchmark_por_filas(A,B,C,m,p,n);

        m = 2048; p = 1024; n = 2048;
        benchmark_por_filas(A,B,C,m,p,n);

        m = 1024; p = 2048; n = 1024;
        benchmark_por_filas(A,B,C,m,p,n);

        // Evaluar tiempo de mult_por_bloques
        unsigned int vector[] = { 64, 128, 256, 512 };
        for (unsigned int i = 0; i < sizeof(vector)/sizeof(vector[0]); i++) {
            nb = vector[i];

            m = 2048; p = 2048; n = 2048;
            benchmark_por_bloques(A,B,C,m,p,n,nb);

            m = 2048; p = 1024; n = 2048;
            benchmark_por_bloques(A,B,C,m,p,n,nb);

            m = 1024; p = 2048; n = 1024;
            benchmark_por_bloques(A,B,C,m,p,n,nb);
        }
    }

    return 0;
}
