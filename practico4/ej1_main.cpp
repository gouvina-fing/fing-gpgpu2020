#include "util.h"
#include "CImg.h"

using namespace cimg_library;

void transpose_global(float * image, int width, int height, float * image_out);
void transpose_shared(float * image, int width, int height, float * image_out);
    
int main(int argc, char** argv){

	const char * path;

	if (argc < 2) printf("Invocar como: './ej1.x nombre_archivo'.\n");
    else
        path = argv[1];

	CImg<float> image(path);
	CImg<float> image_out(image.height(), image.width(),1,1,0);

	float *img_matrix = image.data();
    float *img_out_matrix = image_out.data();
   	
	transpose_global(img_matrix, image.width(), image.height(), img_out_matrix);
   	image_out.save("output_transpose_global.ppm");

	transpose_shared(img_matrix, image.width(), image.height(), img_out_matrix);
   	image_out.save("output_transpose_shared.ppm");
    
	return 0;
}

