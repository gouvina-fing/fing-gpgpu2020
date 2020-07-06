# TODO: Compilar con la flag de nuestra tarjeta (googlear, arch no sé que es para las P100 del cluster)
# nvcc -I/usr/local/cuda/include -o a.out -arch sm_52 -Xcompiler -fopenmp test.cu -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver ????

all:
	nvcc main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas

no_l1:
	nvcc -Xptxas -dlcm=cg main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas