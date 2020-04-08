#include <stdlib.h>
#include <stdio.h>
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

} 

// Ej B) Suma todos los elementos de una matriz recorriendo por columnas
int suma_porcolumnas(int **m, int N) {

} 


int main(int argc, char *argv[]){

   if (argc < 2) {
       printf("No se indico el tamaño \n");
       exit(1);
   }

    unsigned int N = atoi(argv[1]);


    int * vector = (int*) malloc(N*sizeof(int)); 

    srand(0); // Inicializa la semilla aleatoria

    int ** matriz; 
    
    matriz = (int**) malloc(N*sizeof(int*));

    for (int i = 0; i < N; ++i) {
        matriz[i] = (int*) malloc(N*sizeof(int)); 
    }

    random_matriz(matriz,N);

    struct timeval t_i, t_f;

    gettimeofday(&t_i, NULL);
    int resSumaFil = suma_porfilas(matriz,N);
    gettimeofday(&t_f, NULL);
    double t_sgetrf_fila = TIME(t_i,t_f);

    gettimeofday(&t_i, NULL);
    int resSumaCol = suma_porcolumnas(matriz,N); 
    gettimeofday(&t_f, NULL);
    double t_sgetrf_columna = TIME(t_i,t_f);

    printf("Tamanho: %i Resultado sumafil: %i Tiempo fila:    %f ms\n", N, resSumaFil, t_sgetrf_fila);
    printf("Tamanho: %i Resultado sumacol: %i Tiempo columna: %f ms\n", N, resSumaCol, t_sgetrf_columna);

    for (int i = 0; i < N; ++i) {
        free(matriz[i]);
    }
    free(matriz);
    free(vector);

	return 0;
}
