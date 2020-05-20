#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void blur_cpu(float * image_in, int width, int height, float * image_out, float * mask, int m_size);
void blur_gpu(float * image, int width, int height, float * image_out, float * mask, int algorithm);
    
int main(int argc, char** argv){

	const char * path;
	int algorithm;

	if (argc < 3) {
		printf("Invocar como: './ej2.x nombre_archivo, algoritmo'\n");
        printf("-> Algoritmo:\n");
		printf("\t 1 - Kernel con memoria global\n");
		printf("\t 2 - Kernel con memoria compartida\n");
		printf("\t 3 - Kernel con memoria compartida y mascara read_only con restricted pointer\n");
		printf("\t 4 - Kernel con memoria compartida y mascara en memoria constante\n");
		printf("\t 0 - Todos los algoritmos\n");
	} else {
		path = argv[1];
		algorithm = atoi(argv[2]);
	}

    // Inicializamos la mascara. El color del pixel original se conserva (más peso), pero los pixeles vecinos inciden en su valor en menor medida
    float mascara[121]={0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.5, 1  , 0.5, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.5, 2  , 3  , 2  , 0.5, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.5, 2  , 8  , 24 , 8  , 2  , 0.5, 0.1, 0.1,
                        0.1, 0.2, 1  , 3  , 24 , 36 , 24 , 3  , 1  , 0.2, 0.1,
                        0.1, 0.1, 0.5, 2  , 8  , 24 , 8  , 2  , 0.5, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.5, 2  , 3  , 2  , 0.5, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.5, 1  , 0.5, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

	CImg<float> image(path);
	CImg<float> image_out(image.width(), image.height(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();

	switch(algorithm) {
        case 1:
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_global.ppm");
            break;
        case 2:
			blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_a.ppm");
            break;
        case 3:
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_b1.ppm");
            break;
        case 4:
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, algorithm);
   			image_out.save("output_blur_gpu_shared_b2.ppm");
            break;
        case 0:
            blur_cpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 11);
   	        image_out.save("output_blur_cpu.ppm");
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 1);
   			image_out.save("output_blur_gpu_global.ppm");
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 2);
   			image_out.save("output_blur_gpu_shared_a.ppm");
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 3);
   			image_out.save("output_blur_gpu_shared_b1.ppm");
            blur_gpu(img_matrix, image.width(), image.height(), img_out_matrix, mascara, 4);
   			image_out.save("output_blur_gpu_shared_b2.ppm");
            break;
        default:
            printf("Invocar como: './ej2.x nombre_archivo, algoritmo'\n");
            printf("-> Algoritmo:\n");
            printf("\t 1 - Kernel con memoria global\n");
            printf("\t 2 - Kernel con memoria compartida\n");
            printf("\t 3 - Kernel con memoria compartida y mascara read_only con restricted pointer\n");
            printf("\t 4 - Kernel con memoria compartida y mascara en memoria constante\n");
            printf("\t 0 - Todos los algoritmos\n");
    }
   	
	return 0;
}

