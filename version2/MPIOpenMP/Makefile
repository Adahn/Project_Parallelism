target: decide

CFLAGS=-g -O3 -std=c99 -fopenmp
CC=mpicc

POSITION="/5p/4p/4PP/4K///k w"

main.o: main.c projet.h
aux.o: aux.c projet.h


decide: main.o aux.o
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o decide

exec:
	time mpirun -n 2 -x OMP_NUM_THREADS=2 -hostfile hostfile --bynode ./decide $(POSITION)
