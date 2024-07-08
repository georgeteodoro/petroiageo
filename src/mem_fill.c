#include <stdio.h>
#include <stdlib.h>


int main(int argc, char const *argv[])
{
	long size_gb = atoi(argv[1]);

	long num_allocs = 1;
	if (argc == 3) {
		num_allocs = atoi(argv[2]);
	}

	long alloc_size = size_gb * 1024 * 1024 * 1024 / sizeof(long) / num_allocs;

	long *allocs[num_allocs];
	
	printf("alloc size: %ld\n", alloc_size);

	for (long i=0; i<num_allocs; i++) {
		printf("alloc-ing: %ld longs\n", alloc_size);
		allocs[i] = malloc(alloc_size * sizeof(long));
		for (long j=0; j<alloc_size/10; j++){
			allocs[i][j*10] = 1;
		}
	}

	// Sleep forever...
	for (;;) pause();

	return 0;
}
