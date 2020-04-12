#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define TIME(t_i,t_f) ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0) - \
                      ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0);
#define RUNS 10

void random_vector(int *a, int N) {
    for (unsigned int i = 0; i < N; i++) {
        float aleatorio = (float)rand() / (float)RAND_MAX;
        if (aleatorio < 0.5)  
            a[i] = 0;
        else a[i] = 1;
    }
}

void copia_valores_vector_matriz(int *a, int **m, int N) {
    // Solamente se inicializan los valores de la diagonal
    for (unsigned int i = 0; i < N; i++) {
        m[i][i] = a[i];
    }
}

// Ej A) Suma todos los elementos de un vector
int suma_vector(int *a, int N){
    int result = 0;
    for (unsigned int i = 0; i < N; i++) {
        result += + a[i];
    }
    return result;
}

// Ej B) Suma todos los elementos que estan almacenados en la diagonal de una matriz (NxN)
int suma_matriz(int **m, int N){
    int result = 0;
    for (unsigned int i = 0; i < N; i++) {
        result += m[i][i];
    }
    return result;
}

// Aux: Corrida única
void corrida(int N, bool corrida_unica) {

    // Generar vector
    int * vector = (int*) malloc(N*sizeof(int)); 
    srand(0); // Inicializa la semilla aleatoria
    random_vector(vector,N);

    // Generar matriz
    int ** matriz; 
    matriz = (int**) malloc(N*sizeof(int*));
    for (int i = 0; i < N; ++i) {
        matriz[i] = (int*) malloc(N*sizeof(int)); 
    }
    copia_valores_vector_matriz(vector, matriz, N);

    struct timeval t_i, t_f;

    // Evaluar tiempo de suma_vector
    gettimeofday(&t_i, NULL);
    int resVector = suma_vector(vector,N); 
    gettimeofday(&t_f, NULL);
    double t_sgetrf_vector = TIME(t_i,t_f);

    // Evaluar tiempo de suma_matriz
    gettimeofday(&t_i, NULL);
    int resMatriz = suma_matriz(matriz,N);
    gettimeofday(&t_f, NULL);
    double t_sgetrf_matriz = TIME(t_i,t_f);

    if (corrida_unica) {
        printf("Tamano: %i, Resultado suma_vector: %i, Tiempo suma_vector: %f ms\n", N, resVector, t_sgetrf_vector);
        printf("Tamano: %i, Resultado suma_matriz: %i, Tiempo suma_matriz: %f ms\n", N, resMatriz, t_sgetrf_matriz);
    } else {
        printf(",%f,%f", t_sgetrf_vector, t_sgetrf_matriz);
    }

    // Liberar memoria
    for (int i = 0; i < N; ++i) {
        free(matriz[i]);
    }
    free(matriz);
    free(vector);
}

int main(int argc, char *argv[]){

    // Leer parametros
    bool corrida_unica = argc > 1;
 
    if (corrida_unica) {
        corrida(atoi(argv[1]), corrida_unica);
    } else {
        unsigned int vector[] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
        for (unsigned int i = 0; i < sizeof(vector)/sizeof(vector[0]); i++) {
            printf("%i", vector[i]);
            for (unsigned int j = 0; j < RUNS; j++) {
                corrida(vector[i], corrida_unica);
            }
            printf("\n");
        }
    }

    return 0;
}
