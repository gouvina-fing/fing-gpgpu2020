#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

// Cifrado: E(x) = (Ax + B) mod M
// Descifrado: D(x) = A^{-1}(x - B) mod M

// A y B son las claves del cifrado. A y M son co-primos.
#define A 15
#define B 27
#define M 256 // Cantidad de caracteres en la tabla ASCII extendida
#define A_MMI_M -17 // A^{-1}. Inverso multiplicativo de A modulo M

// El operador mod no es el mismo que ( %) para números negativos, por lo que se provee una función módulo en el código.
__device__ int modulo(int a, int b) {
    int r = a % b;
    r = (r < 0) ? r + b : r;
    return r;
}

// TODO: Implementar el kernel decrypt_kernel utilizando un solo bloque de threads (en la dimensión x).
//		 A, B y M están definidas en el código como macros.
__global__ void decrypt_kernel(int *d_message, int length) {

}

/*
.  Como cada caracter puede ser encriptado y desencriptado de forma independiente podemos utilizar la
.  GPU para desencriptar el texto en paralelo. Para esto debemos lanzar un kernel que realice el desencriptado,
.  asignando un thread por caracter
.
.  Para resolver los ejercicios deberá utilizar las variables especiales:
.  threadIdx.x, blockIdx.x, blockDim.x y gridDim.x
*/ 
int main(int argc, char *argv[]) {
    int *h_message;
    int *d_message;
    unsigned int size;
    int i;

    const char * fname;

    if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
    else
        fname = argv[1];

    int length = get_text_length(fname);

    size = length * sizeof(int);

    // Reservo memoria para el mensaje
    h_message = (int *)malloc(size);

    // Leo el archivo de la entrada
    read_file(fname, h_message);

    // Reservo memoria en la GPU

    // Copio los datos a la memoria de la GPU

    /* Configuro la grilla de threads
    .  Ej1: 1 bloque de n threads (tamaño de bloque a elección).
    .  Ej2: Varios bloques, procesando textos de largo arbitrario.
    .  Ej3: Cantidad fija de bloques (ej 128) para procesar textos de largo arbitrario.
    */

    // Ejecuto el kernel

    // Copio los datos nuevamente a la memoria de la CPU

    // Despliego el mensaje
    for (int i = 0; i < length; i++) {
        printf("%c", (char)h_message[i]);
    }
    printf("\n");

    // Libero la memoria en la GPU

    // Libero la memoria en la CPU
    free(h_message);

    return 0;
}

    
int get_text_length(const char * fname) {
    FILE *f = NULL;
    f = fopen(fname, "r"); // read and binary flags

    size_t pos = ftell(f);    
    fseek(f, 0, SEEK_END);    
    size_t length = ftell(f); 
    fseek(f, pos, SEEK_SET);  

    fclose(f);

    return length;
}

void read_file(const char * fname, int* input) {
    // printf("leyendo archivo %s\n", fname );

    FILE *f = NULL;
    f = fopen(fname, "r"); // read and binary flags
    if (f == NULL){
        fprintf(stderr, "Error: Could not find %s file \n", fname);
        exit(1);
    }

    // fread(input, 1, N, f);
    int c; 
    while ((c = getc(f)) != EOF) {
        *(input++) = c;
    }

    fclose(f);
}
