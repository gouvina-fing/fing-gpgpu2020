#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void blur_cpu(float * image_in, int width, int height, float * image_out, float * mask, int m_size);
void blur_gpu(float * image, int width, int height, float * image_out, float * mask, int algorithm);
    
int main(int argc, char** argv){

	const char * path;
	int algorithm;

	if (argc < 3) {
		printf("Invocar como: './blur.x nombre_archivo, algoritmo'. En donde algoritmo es:");
		printf("\t 0 - Práctico 3) Kernel con memoria global");
		printf("\t 1 - Ej 2a) Kernel con memoria compartida");
		printf("\t 2 - Ej 2b1) Kernel con memoria compartida y");
		printf("\t 3 - Ej 2b2) Kernel con con memoria compartida y ");
		printf("\t 4 - Todos los algoritmos");
		printf("\n");
	} else {
		path = argv[1];
		algorithm = atoi(argv[2]);
	}

    // Inicializamos la mascara. El color del pixel original se conserva (más peso), pero los pixeles vecinos inciden en su valor en menor medida
    float mascara[25]={1, 4, 6, 4, 1,
                       4,16,24,16, 4,
                       6,24,36,24, 6,
                       4,16,24,16, 4,
                       1, 4, 6, 4, 1};

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();

	float elapsed = 0;

	blur_cpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 5);
   	image_out.save("output_blur_cpu.ppm");

	switch(algorithm) {
        // Práctico 3) Kernel con memoria global
        case 0:
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_global.ppm");
            break;
        // Ej 2a) Kernel con memoria compartida
        case 1:
            
			blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_a.ppm");
            break;
        // Ej 2b1) Kernel con memoria compartida y optimizando la máscara cómo read_only y restricted pointer
        case 2:
			
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_b1.ppm");
            break;
        // Ej 2b2) Kernel con con memoria compartida y almacenando la máscara en la memoria constante de la GPU
        case 3:
			
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_b2.ppm");
            break;
        default:
            printf("Algoritmo seleccionado invalido.\n");
            printf("Invocar como: './ej2.x nombre_archivo, algoritmo'. En donde algoritmo es:\n");
            printf("\t 0 - Práctico 3) Kernel con memoria global\n");
            printf("\t 1 - Ej 2a) Kernel con memoria compartida\n");
            printf("\t 2 - Ej 2b1) Kernel con memoria compartida y optimizando la máscara cómo read_only y restricted pointer\n");
            printf("\t 3 - Ej 2b2) Kernel con con memoria compartida y almacenando la máscara en la memoria constante de la GPU\n");
            printf("\n");
    }
   	
	return 0;
}

