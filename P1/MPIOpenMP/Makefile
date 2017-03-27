target: decide

CFLAGS=-fopenmp -g -O3 -std=c99
CC=mpicc

POSITION="///2kpK/7P w"

main.o: main.c projet.h
aux.o: aux.c projet.h


decide: main.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o decide

exec:
	time mpirun -x OMP_NUM_THREADS=4 -n 4 -hostfile hostfile --bynode ./decide $(POSITION)
