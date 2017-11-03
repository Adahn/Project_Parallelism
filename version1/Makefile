DIR=OpenMP MPI MPIOpenMP

.PHONY: all $(DIR)

dir:$(DIR)

$(DIR):
	$(MAKE) -C $@

clean:
	for directory in $(DIR); do \
		$(MAKE) -C $$directory clean; \
	done

