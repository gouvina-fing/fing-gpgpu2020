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

// TODO: Implementar el kernel decrypt_kernel utilizando un solo bloque de threads (en la dimensión x).
//		 A, B y M están definidas en el código como macros.
// Para resolver los ejercicios deberá utilizar las variables especiales: threadIdx.x, blockIdx.x, blockDim.x y gridDim.x
__global__ void decrypt_kernel(int *device_message, int length) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < length) {
        device_message[index] = modulo(A_MMI_M*(device_message[index] - B), M);
    }
}

/*
.  Normalmente: For recorriendo todo el texto usando D(x)
.  Como cada caracter puede ser encriptado y desencriptado de forma independiente podemos utilizar la
.  GPU para desencriptar el texto en paralelo. Para esto debemos lanzar un kernel que realice el desencriptado,
.  asignando un thread por caracter
*/ 
int main(int argc, char *argv[]) {
    int *host_message;
    int *device_message;
    unsigned int size;
    int i;

    const char * fname;

    if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
    else
        fname = argv[1];

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

    /* Configuro la grilla de threads
    .  Ej1: 1 bloque de n threads (tamaño de bloque a elección nuestra más de 1024 supera el límite de CUDA).
    .       Limitante que tendremos, ejecutando un solo bloque de 1024 threads como máximo podremos desencriptar solo 1024 caracteres
    .  Ej2: Varios bloques, procesando textos de largo arbitrario.
    .       Extendemos la parte 1 para que el kernel pueda usar varios bloques (tantos como sea necesario para desencriptar todo el texto)
    .       En cuda podemos ejecutar hasta millones de bloques en la dimensión x (esto es unidimensional)
    .       Definimos la cantidad de bloques a usar en base al largo del texto (Esta es la forma que se trabaja en realidad)
    .  Ej3: Cantidad fija de bloques (ej 128) para procesar textos de largo arbitrario.
    .       Se fijan los bloques como limitante. Tendremos tantos hilos como 128 * cantHilosPorBloque
    .       Por lo que para procesar textos largos habrá que hacer algo secuencial, haciendo que cada hilo desencripte más de un caracter.
    .       Esto no es ideal en entorno cuda, es suboptimo a la parte 2 y al principio de que el codigo a ejecutar sea simple, pero este ejercicio es puramente didactico. 
    */
    int block_size = 1024;
    // Ej 1: (no procesa más de 1024 caracteres)
    // int bloques = 1
    // Ej2:
    int bloques = length/block_size + (length % block_size != 0); // Division with ceiling
    
    // TODO: Cómo es esto de dim3? (tipo ya sé que es, y la usan en las ppts, tiene sentido usarla para arrays, no encontré ejemplos de arrays unidimensionales que lo usen)
    dim3 tamGrid(bloques, 1); // Grid dimension
    dim3 tamBlock(block_size, 1, 1); // Block dimension

    // Ejecuto el kernel
    decrypt_kernel<<<tamGrid, tamBlock>>>(device_message, length);

    // Copio los datos nuevamente a la memoria de la CPU
    CUDA_CHK(cudaMemcpy(host_message, device_message, size, cudaMemcpyDeviceToHost)); // puntero destino, puntero origen, numero de bytes a copiar, tipo de transferencia

    // Despliego el mensaje
    for (int i = 0; i < length; i++) {
        printf("%c", (char)host_message[i]);
    }
    printf("\n");

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
