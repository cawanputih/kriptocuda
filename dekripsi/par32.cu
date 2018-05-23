#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 20480;
int dimensigrid = 20;
int dimensiblok = 1024;

__host__ __device__ void modexp(ulint a, ulint b, ulint c, ulint* res) {
	ulint64 s = a;
	ulint64 ans = 1;
	while (b != 0) {
		if (b % 2 == 1) {
			ans = ans * s % c;
			b--;
		}
		b /= 2;
		if (b != 0) {
			s = s * s %c;
		}
	}
	*res = ans;
}

__device__ void dekripsi(ulint a, ulint b, ulint p, ulint e, ulint *res) {
	modexp(a, e, p, res);
	*res = *res * b % p;
}

__global__ void kerneldek(ulint *c, ulint p, ulint e, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	dekripsi(c[2*i], c[2*i+1], p, e, res + i);
}

void dekripsiCUDA(ulint *c, ulint p, ulint e, ulint *res) {
	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devc, *devres;
	
	cudaMalloc((void**)&devc, 2 * banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres, banyakdata * sizeof(ulint));
	
	cudaMemcpy((devc), c, (sizeof(ulint) * banyakdata *2), cudaMemcpyHostToDevice);

	kerneldek << <dimensigrid, dimensiblok>> >(devc,p,e,devres);

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res, devres, (sizeof(ulint) * banyakdata), cudaMemcpyDeviceToHost);
	
	cudaFree(devc);
	cudaFree(devres);
}

void init(ulint *c){
	
	for (int i = 0; i < 2 * banyakdata; i++) {
		c[i] = rand() % 3999999978;
	}	
}

int main(){
	ulint *c, *res, p, e;

	c = (ulint*)malloc(2 * banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * sizeof(ulint));

	srand(2018);

	p = 3999999979;
	e = rand() % 3999999978;
	init(c);

	dekripsiCUDA(c,p,e,res);


	// printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("c[%d] = %lu 	c[%d] = %lu\n", 2*i, res[2*i], 2*i+1, res[2*i+1]);
	// }

	// printf("c ...\n");
	// printf("c[%d] = %lu 	c[%d] = %lu\n", banyakdata * 2-2, res[banyakdata * 2-2], banyakdata *2-1,res[banyakdata*2-1]);

	// printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("m[%d] = %lu\n", i, res[i]);
	// }

	// printf("m[...]\n");
	// printf("m[%d] = %lu\n", banyakdata-1, res[banyakdata-1]);

	free(c);
	free(res);

	return 0;
}
