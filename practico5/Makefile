# -arch=sm_75 Turing GTX 1660 y GTX 1660S
# -arch=sm_60 Pascal P-100

all:
	nvcc main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas -arch=sm_75

no_l1:
	nvcc -Xptxas -dlcm=cg main.cpp dgemm.cu dtrsm.cu -o labgpu20_no_l1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas -arch=sm_75

cluster:
	nvcc main.cpp dgemm.cu dtrsm.cu -o labgpu20.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas -arch=sm_60

cluster_no_l1:
	nvcc -Xptxas -dlcm=cg main.cpp dgemm.cu dtrsm.cu -o labgpu20_no_l1.x -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lcublas -arch=sm_60
