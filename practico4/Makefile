all:
	nvcc ej2_main.cpp blur.cu -o ej2.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
	#nvcc -Xptxas -dlcm=cg ej2_main.cpp blur.cu -o ej2_no_l1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11