CC = gcc

# Compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS = -g

TARGET = program

all: Ej1.o Ej2.o Ej3.o
	$(CC) $(CFLAGS) -o main Ej1.o Ej2.o Ej3.o

Ej1.o: Ej1.c
	$(CC) $(CFLAGS) -c Ej1.c

Ej2.o: Ej2.c
	$(CC) $(CFLAGS) -c Ej2.c

Ej3.o: Ej3.c
	$(CC) $(CFLAGS) -c Ej3.c

clean: 
	$(RM) program *.o .~
