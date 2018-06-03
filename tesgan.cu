#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int banyakdata = 327680;
int dimensigrid = 320;
int dimensiblok = 1024;


__global__ void Kernel(int* m)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < 1000000; i++) {
		int a = i * i;
		int b = a * i;
		m[idx] = a-b;
	}
}

void kernel2(int *m)
{
	for (int idx = 0; idx < banyakdata; idx++)
	{
		for (int i = 0; i < 1000000; i++) {
			int a = i * i;
			int b = a * i;
			m[idx] = a-b;
		}
	}
}

void paralel (int *m)
{
	int *devm;
	
	cudaMalloc((void**)&devm, banyakdata * sizeof(int));
	cudaMemcpy(devm, m, (sizeof(int) * banyakdata), cudaMemcpyHostToDevice);
	
	Kernel << <dimensigrid, dimensiblok>> >(devm);

	cudaMemcpy(m, devm, (sizeof(int) * banyakdata), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
}

void sekuensial(int *m) {
	clock_t begin = clock();
		kernel2(m);
	clock_t end = clock();
	double time_spent = (double)(end - begin);
	printf("Durasi = %f milli seconds\n", time_spent);
}


int main()
{
	srand(2018);
	int *m = (int*) malloc(banyakdata * sizeof(int));
	for (int i = 0; i < banyakdata; i++)
	{
		m[i] = rand() % 200000;
	}

	//paralel(m);
	sekuensial(m);

	return 0;
}