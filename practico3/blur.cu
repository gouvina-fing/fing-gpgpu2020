#include "util.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// Macro para wrappear funciones de cuda e interceptar errores
#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Ej 2a) Kernel que aplica el filtro Gaussiano en la GPU
__global__ void blur_kernel(float* d_input, int width, int height, float* d_output, float* d_msk, int m_size) {

}

// Ej 1a) Threads con índice consecutivo en la dirección x deben acceder a pixels de una misma fila de la imagen.
//        Es importante usar blockIdx y threadIdx adecuadamente para acceder a la estructura bidimensional.
__global__ void ajustar_brillo_coalesced_kernel(float* d_input, float* d_output, int width, int height, float coef) {

}

// Ej 1a) Threads con índice consecutivo en la dirección x deben acceder a pixels de una misma columna de la imagen.
//        Es importante usar blockIdx y threadIdx adecuadamente para acceder a la estructura bidimensional.
__global__ void ajustar_brillo_no_coalesced_kernel(float* d_input, float* d_output, int width, int height, float coef) {

}

// Procesa la img en GPU sumando un coeficiente entre -255 y 255 a cada píxel, aumentando o reduciendo su brillo.
void ajustar_brillo_gpu(float * img_in, int width, int height, float * img_out, float coef, int filas=1) {
    
    // Reservar memoria en la GPU

    // Copiar imagen y máscara a la GPU
   
    // Configurar grilla y lanzar kernel
    // TODO: La grilla (bidimensional) de threads debe estar configurada para aceptar matrices de cualquier tamaño.
    
    // Transferir resultado a la memoria principal

    // TODO: Ej 1b) Registrar tiempos de cada etapa de ajustar_brillo_gpu para las dos variantes. Discutir diferencia entre variantes.
    //              (tiempos, reserva de memoria, transferencia de datos, ejecución del kernel, etc)
    //              Usar util.h

    // TODO: Ej 1c) Compare los resultados de la salidad de nvprof.
    //              Registrar con nvprof --profileapi-trace none --metrics gld_efficiency ./blur imagen.ppm
    //              Qué puede decir del resultado de la métrica gld_efficiency?
    //              Duda: Esto se hace acá o en main.cpp?

    // Liberar la memoria
}

// Ej 2) Aplica un filtro Gaussiano que reduce el ruido de una imagen en escala de grises.
//       El filtro sustituye el valor de intensidad de cada pixel por un promedio ponderado de los pixeles vecinos.
//       Los pesos por los cuales se pondera cada vecino en el promedio se almacenan en una matriz cuadrada (máscara)
void blur_gpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size){
    
    // Reservar memoria en la GPU

    // Copiar imagen y máscara a la GPU
   
    // Configurar grilla y lanzar kernel
    // TODO: La grilla (bidimensional) de threads debe estar configurada para aceptar matrices de cualquier tamaño.
    // Es importante en el kernel usar blockIdx y threadIdx adecuadamente para acceder a esta estructura.

    // Transferir resultado a la memoria principal

    // TODO: Ej 2b) Registre los tiempos de cada etapa de la función y compare las variantes de CPU y GPU.
    //              ¿Qué aceleración se logra? ¿Y considerando únicamente el tiempo del kernel (cudaMemcpy tiene mucho overhead!)?
    //              Duda: Esto se hace acá o en main.cpp?

	// Liberar la memoria
}

// Recorre la imagen sumando secuencialmente un coeficiente entre -255 y 255 a cada píxel, aumentando o reduciendo su brillo.
void ajustar_brillo_cpu(float * img_in, int width, int height, float * img_out, float coef) {

    CLK_POSIX_INIT;
    CLK_POSIX_START;

    for(int imgx=0; imgx < width ; imgx++) {
        for(int imgy=0; imgy < height; imgy++) {
            img_out[imgy*width+imgx] = min(255.0f,max(0.0f,img_in[imgy*width+imgx]+coef));
        }
    }

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    printf("Tiempo ajustar brillo CPU: %f ms\n", t_elap);
}

// Recorre la imagen aplicando secuencialmente un filtro Gaussiano que reduce el ruido de una imagen en escala de grises.
void blur_cpu(float * img_in, int width, int height, float * img_out, float msk[], int m_size) {

    CLK_POSIX_INIT;
    CLK_POSIX_START;

    float val_pixel=0;
    
    // Para cada pixel aplicamos el filtro
    for(int imgx=0; imgx < width ; imgx++) {
        for(int imgy=0; imgy < height; imgy++) {

            val_pixel = 0;

            // Aca aplicamos la mascara
            for (int i = 0; i < m_size ; i++) {
                for (int j = 0; j < m_size ; j++) {
                    
                    int ix =imgx + i - m_size/2;
                    int iy =imgy + j - m_size/2;
                    
                    if(ix >= 0 && ix < width && iy>= 0 && iy < height )
                        val_pixel = val_pixel +  img_in[iy * width +ix] * msk[i*m_size+j];
                }
            }
            
            // Guardo valor resultado
            img_out[imgy*width+imgx]= val_pixel;
        }
    }

    CLK_POSIX_STOP;
    CLK_POSIX_ELAPSED;

    printf("Tiempo filtro Gaussiano CPU: %f ms\n", t_elap);
}