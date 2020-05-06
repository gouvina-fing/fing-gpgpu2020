all:
	nvcc main.cpp blur.cu -o blur -O3 -L/usr/X11R6/lib -lm -lpthread -lX11
	nvcc -Xptxas -dlcm=cg main.cpp blur.cu -o blur_no_cache_l1 -O3 -L/usr/X11R6/lib -lm -lpthread -lX11