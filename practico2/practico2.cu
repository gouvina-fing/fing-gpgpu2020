#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Macro para wrappear funciones de cuda e interceptar errores
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

// El operador mod no es el mismo que (%) para números negativos, por lo que se provee una función módulo en el código.
__device__ int modulo(int a, int b) {
    int r = a % b;
    r = (r < 0) ? r + b : r;
    return r;
}

// Kernel para el ejercicio 1 y 2. Cada hilo procesa un solo caracter.
__global__ void decrypt_kernel(int *device_message, int length) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < length) {
        device_message[index] = modulo(A_MMI_M*(device_message[index] - B), M);
    }
}

// Kernel para el ejercicio 3. Cada hilo procesa char_per_block caracteres.
// Para mantener el coalesced memory access cada cada warp de hilos procesa caracteres secuenciales.
// Es por esto que un hilo modifica caracteres no contiguos (separados por una distancia block_span).
// En lugar de su index sea index*char_per_block y cada hilo modifique char_per_block caracteres contiguos (lo cual no mantendría el coalesced memory access).
__global__ void decrypt_kernel_ej3(int *device_message, int char_per_block, int length) {
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;
    int block_span = blockDim.x*gridDim.x;
    for(int i = 0; i < char_per_block; ++i) {
        if((index + i*block_span) < length) {
            device_message[index + i*block_span] = modulo(A_MMI_M*(device_message[index + i*block_span] - B), M);
        }
    }
}

__global__ void old_ecrypt_kernel_ej3(int *device_message, int char_per_block, int length) {
    int index = (blockIdx.x*blockDim.x + threadIdx.x)*char_per_block;
    for(int i = char_per_block; i > 0; --i) {
        if(index - i < length) {
            device_message[index - i] = modulo(A_MMI_M*(device_message[index - i] - B), M);
        }
    }
}

/*
.  Normalmente se resolvería con un for recorriendo todo el texto usando D(x)
.  Como cada caracter puede ser encriptado y desencriptado de forma independiente podemos utilizar la GPU para desencriptar el texto en paralelo.
.  Para esto debemos lanzar un kernel que realice el desencriptado, asignando un thread por caracter.
*/ 
int main(int argc, char *argv[]) {
    int *host_message;
    int *device_message;
    unsigned int size;

    const char * fname;
    int algorithm;

    if (argc < 3) printf("Invocar como: './practico_sol nombre_archivo, ejercicio'. En donde ejercicio es 1, 2 o 3.\n");
    else
        fname = argv[1];
        algorithm = atoi(argv[2]);

    int length = get_text_length(fname);

    size = length * sizeof(int);

    // Reservo memoria para el mensaje
    host_message = (int *)malloc(size);

    // Leo el archivo de la entrada
    read_file(fname, host_message);

    // Reservo memoria en la GPU
    CUDA_CHK(cudaMalloc((void**)& device_message, size));

    // Copio los datos a la memoria de la GPU
    CUDA_CHK(cudaMemcpy(device_message, host_message, size, cudaMemcpyHostToDevice)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia

    int block_size = 1024;
    int char_per_block, bloques;

    switch(algorithm) {
        // Ej1: 1 bloque de 1024 threads (No procesa más de 1024 caracteres).
        case 1:
            bloques = 1;
            char_per_block = 1;
            break;
        
        // Ej2: Múltiples bloques, los necesarios para procesar todo el texto.
        case 2:
            bloques = length/block_size + (length % block_size != 0); // Division with ceiling
            char_per_block = 1;
            break;

        // Ej3: 128 bloques. Procesa todo el texto
        //      Si el texto tiene más caracteres de 128*1024 un kernel debe procesar más de un caracter secuencialmente.
        case 3:
            bloques = 128;
            char_per_block = length/(bloques*block_size) + (length % bloques*block_size != 0);
            break;
        default:
            printf("Algoritmo seleccionado invalido.\n");
            printf("Invocar como: './practico_sol nombre_archivo, ejercicio'. En donde ejercicio es 1, 2 o 3.\n");
    }
    
    dim3 tamGrid(bloques); // Grid dimension
    dim3 tamBlock(block_size); // Block dimension

    // Ejecuto el kernel
    if(algorithm == 1 || algorithm == 2) {
        decrypt_kernel<<<tamGrid, tamBlock>>>(device_message, length);
    } else {
        if(algorithm == 3) {
            decrypt_kernel_ej3<<<tamGrid, tamBlock>>>(device_message, char_per_block, length);
        }
    }

    // Copio los datos nuevamente a la memoria de la CPU
    CUDA_CHK(cudaMemcpy(host_message, device_message, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia

    // Despliego el mensaje
    if(algorithm == 1 || algorithm == 2 || algorithm == 3) {
        for (int i = 0; i < length; i++) {
            printf("%c", (char)host_message[i]);
        }
        printf("\n");
    }
    
    // Libero la memoria en la GPU
    CUDA_CHK(cudaFree(device_message));

    // Libero la memoria en la CPU
    free(host_message);

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
