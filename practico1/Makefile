CC = gcc

# Compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS = -g -O3

TARGET = program

all: ej1 ej2 ej3

ej1: ej1.c
	$(CC) $(CFLAGS) -o ej1.x ej1.c

ej2: ej2.c
	$(CC) $(CFLAGS) -o ej2.x ej2.c

ej3: ej3.c
	$(CC) $(CFLAGS) -o ej3.x ej3.c

clean: 
	$(RM) *.o *.x

clean_data:
	$(RM) *.csv