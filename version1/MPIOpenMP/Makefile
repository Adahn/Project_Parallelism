target: decide

CFLAGS=-g -O3 -std=c99 -fopenmp
CC=mpicc

POSITION="4k//4K/4P w"

main.o: main.c projet.h
aux.o: aux.c projet.h


decide: main.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o decide

exec:
	time mpirun -n 2 -x OMP_NUM_THREADS=2 -x OMP_SCHEDULE="dynamic" -hostfile hostfile --bynode ./decide $(POSITION)
