#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned char ulint;
typedef unsigned int ulint64;

int banyakdata = 335544320;
int dimensigrid = 327680;
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

__device__ void enkripsi(ulint g, ulint k, ulint p, ulint m, ulint y, ulint *res) {
	modexp(g, k, p, res);
	modexp(y, k, p, res + 1);
	
	*(res + 1) = *(res + 1) * m % p;
}

__device__ void dekripsi(ulint a, ulint b, ulint p, ulint e, ulint *res) {
	modexp(a, e, p, res);
	*res = *res * b % p;
}

__global__ void kernelenk(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	enkripsi(g, k[i], p, m[i], y, res + 2 * i);
}

__global__ void kerneldek(ulint *c, ulint p, ulint e, ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	dekripsi(c[2*i], c[2*i+1], p, e, res + i);
}

void enkripsiCUDA(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devm, *devk, *devres;
	
	cudaMalloc((void**)&devm, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devk, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres, banyakdata * 2 * sizeof(ulint));
	
	cudaMemcpy((devm), m, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);
	cudaMemcpy((devk), k, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);	

	kernelenk << <dimensigrid, dimensiblok>> >(devm,devk,g,p,y,devres);

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res, devres, (sizeof(ulint) * 2 * banyakdata), cudaMemcpyDeviceToHost);
	
	cudaFree(devm);
	cudaFree(devk);
	cudaFree(devres);
}

void initenkripsi(ulint *m, ulint *k){
	
	for (int i = 0; i < banyakdata; i++) {
		m[i] = rand() % 250;
		k[i] = rand() % 250;
	}	
}

int main(){
	ulint *m, *k, *res, g, p, y, x;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	k = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * 2 * sizeof(ulint));

	srand(2018);

	g = rand() % 250;
	p = 251;
	x = rand() % 250;
	modexp(g,x,p,&y);
	initenkripsi(m, k);

	enkripsiCUDA(m,k,g,p,y,res);


	// printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("c[%d] = %d 	c[%d] = %d\n", 2*i, res[2*i], 2*i+1, res[2*i+1]);
	// }

	// printf("c ...\n");
	// printf("c[%d] = %d 	c[%d] = %d\n", banyakdata * 2-2, res[banyakdata * 2-2], banyakdata *2-1,res[banyakdata*2-1]);

	free(m);
	free(k);
	free(res);

	return 0;
}
