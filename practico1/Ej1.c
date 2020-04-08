#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define TIME(t_i,t_f) ((double) t_f.tv_sec * 1000.0 + (double) t_f.tv_usec / 1000.0) - \
                      ((double) t_i.tv_sec * 1000.0 + (double) t_i.tv_usec / 1000.0);

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

}

// Ej B) Suma todos los elementos que estan almacenados en la diagonal de una matriz (NxN)
int suma_matriz(int **m, int N){

}


int main(int argc, char *argv[]){

   if (argc < 2) {
       printf("No se indico el tamaño \n");
       exit(1);
   }

    unsigned int N = atoi(argv[1]);


    int * vector = (int*) malloc(N*sizeof(int)); 

    srand(0); // Inicializa la semilla aleatoria
    random_vector(vector,N);

    int ** matriz; 
    
    matriz = (int**) malloc(N*sizeof(int*));

    for (int i = 0; i < N; ++i)
    {
        matriz[i] = (int*) malloc(N*sizeof(int)); 
    }

    copia_valores_vector_matriz(vector,matriz,N);

    struct timeval t_i, t_f;

    gettimeofday(&t_i, NULL);
    int resVector = suma_vector(vector,N); 
    gettimeofday(&t_f, NULL);
    double t_sgetrf_vector = TIME(t_i,t_f);

    gettimeofday(&t_i, NULL);
    int resMatriz = suma_matriz(matriz,N);
    gettimeofday(&t_f, NULL);
    double t_sgetrf_matriz = TIME(t_i,t_f);

    printf("Tamanho: %i Resultado sumavec: %i Tiempo vector: %f ms\n", N, resVector, t_sgetrf_vector);
    printf("Tamanho: %i Resultado sumadiag: %i Tiempo matriz: %f ms\n", N, resMatriz, t_sgetrf_matriz);

    for (int i = 0; i < N; ++i)
    {
        free(matriz[i]);
    }
    free(matriz);
    free(vector);

    return 0;
}
