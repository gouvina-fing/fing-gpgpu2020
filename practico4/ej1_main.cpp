#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void transpose_gpu(float * image, int width, int height, float * image_out, int algorithm);
    
int main(int argc, char** argv){

	const char * path;
	int algorithm;

	if (argc < 3) {
		printf("Invocar como: './ej1.x nombre_archivo, algoritmo'\n");
        printf("-> Algoritmo:\n");
		printf("\t 1 - Kernel con memoria global\n");
		printf("\t 2 - Kernel con memoria compartida\n");
		printf("\t 3 - Kernel con memoria compartida y columna extra\n");
		printf("\t 0 - Todos los algoritmos\n");
	} else {
		path = argv[1];
		algorithm = atoi(argv[2]);
	}

	CImg<float> image(path);
	CImg<float> image_out(image.height(), image.width(), 1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();

	switch(algorithm) {
        case 1:
            transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, algorithm);
   			image_out.save("output_transpose_global.pgm");
            break;
        case 2:
            transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, algorithm);
   			image_out.save("output_transpose_shared.pgm");
            break;
        case 3:
            transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, algorithm);
			image_out.save("output_transpose_shared_extra.pgm");
            break;
        case 0:
            transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, 1);
   			image_out.save("output_transpose_global.pgm");
			transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, 2);
   			image_out.save("output_transpose_shared.pgm");
			transpose_gpu(img_matrix, image.width(), image.height(), img_out_matrix, 3);
   			image_out.save("output_transpose_shared_extra.pgm");
            break;
        default:
            printf("Invocar como: './ej1.x nombre_archivo, algoritmo'\n");
			printf("-> Algoritmo:\n");
			printf("\t 1 - Kernel con memoria global\n");
			printf("\t 2 - Kernel con memoria compartida\n");
			printf("\t 3 - Kernel con memoria compartida y columna extra\n");
			printf("\t 0 - Todos los algoritmos\n");
    }
    
	return 0;
}

