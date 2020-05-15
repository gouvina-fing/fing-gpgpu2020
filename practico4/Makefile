all:
	nvcc ej1_main.cpp transpose.cu -o ej1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
	nvcc ej2_main.cpp blur.cu -o ej2.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
	#nvcc -Xptxas -dlcm=cg ej2_main.cpp blur.cu -o ej2_no_l1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11

ej1:
	nvcc ej1_main.cpp transpose.cu -o ej1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11

ej2:
	nvcc ej2_main.cpp blur.cu -o ej2.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11