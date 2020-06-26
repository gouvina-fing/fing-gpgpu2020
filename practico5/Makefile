# TODO: Compilar con la flag de nuestra tarjeta (googlear, arch no sé que es para las P100 del cluster)

all:
	nvcc main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11

no_l1:
	nvcc -Xptxas -dlcm=cg main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11