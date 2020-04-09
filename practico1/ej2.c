#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define TIME(t_i,t_f) ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0) - \
                      ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0);

void random_matriz(int **m, int N) {
    for (unsigned int i = 0; i < N; i++)
        for (unsigned int j = 0; j < N; j++) { 
            float aleatorio = (float)rand() / (float)RAND_MAX;
            if (aleatorio < 0.5)  
                m[i][j] = 0;
            else m[i][j] = 1;
        }
}

// Ej A) Suma todos los elementos de una matriz recorriendo por filas
int suma_porfilas(int **m, int N) {
    int res = 0;
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            res = res + m[i][j];
        }
    }
    return res;
} 

// Ej B) Suma todos los elementos de una matriz recorriendo por columnas
int suma_porcolumnas(int **m, int N) {
    int res = 0;
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            res = res + m[j][i];
        }
    }
    return res;
}

// Aux: Corrida única
void corrida(int N, bool corrida_unica) {

    // Generar matriz
    srand(0); // Inicializa la semilla aleatoria
    int ** matriz; 
    matriz = (int**) malloc(N*sizeof(int*));
    for (int i = 0; i < N; ++i) {
        matriz[i] = (int*) malloc(N*sizeof(int)); 
    }
    random_matriz(matriz,N);

    struct timeval t_i, t_f;

    // Evaluar tiempo de suma_porfilas
    gettimeofday(&t_i, NULL);
    int resSumaFil = suma_porfilas(matriz,N);
    gettimeofday(&t_f, NULL);
    double t_sgetrf_fila = TIME(t_i,t_f);

    // Evaluar tiempo de suma_porcolumnas
    gettimeofday(&t_i, NULL);
    int resSumaCol = suma_porcolumnas(matriz,N); 
    gettimeofday(&t_f, NULL);
    double t_sgetrf_columna = TIME(t_i,t_f);

    if (corrida_unica) {
        printf("Tamano: %i, Resultado suma_filas: %i, Tiempo suma_filas:    %f ms\n", N, resSumaFil, t_sgetrf_fila);
        printf("Tamano: %i, Resultado suma_columnas: %i, Tiempo suma_columnas: %f ms\n", N, resSumaCol, t_sgetrf_columna);
    } else {
        printf("%i,%f,%f\n", N, t_sgetrf_fila, t_sgetrf_columna);
    }

    // Liberar memoria
    for (int i = 0; i < N; ++i) {
        free(matriz[i]);
    }
    free(matriz);
}


int main(int argc, char *argv[]){

    // Leer parametros
    bool corrida_unica = argc > 1;
 
    if (corrida_unica) {
        corrida(atoi(argv[1]), corrida_unica);
    } else {
        unsigned int vector[] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
        for (unsigned int i = 0; i < sizeof(vector)/sizeof(vector[0]); i++) {
            corrida(vector[i], corrida_unica);
        }
    }

    return 0;
}
