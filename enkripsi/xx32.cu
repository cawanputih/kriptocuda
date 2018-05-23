#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 2048;
int dimensigrid = 2;
int dimensiblok = 1024;

__host__ __device__ ulint modexp(ulint a, ulint b, ulint c) {
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
	return ans;
}


__global__ void kernelenk(ulint *res) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	ulint ka = 77;
	ulint ga = 2;
	ulint pa = 2357;

	res[i] = modexp(ga, ka, pa);
	// enkripsi(g, k[i], p, m[i], y, res + 2 * i);
}

void enkripsiCUDA(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devres;
	
	
	cudaMalloc((void**)&devres, banyakdata * 2 * sizeof(ulint));

	kernelenk << <dimensigrid, dimensiblok>> >(devres);

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res, devres, (sizeof(ulint) * 2 * banyakdata), cudaMemcpyDeviceToHost);
	
	
	cudaFree(devres);
}

void initenkripsi(ulint *m, ulint *k){
	
	for (int i = 0; i < banyakdata; i++) {
		m[i] = 1001;
		k[i] = 77;
	}	
}

int main(){
	ulint *m, *k, *res, g, p, y, x;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	k = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * 2 * sizeof(ulint));

	srand(2018);

	g = 2;
	p = 2357;
	x = 1751;
	y = modexp(g,x,p);
	initenkripsi(m, k);

	enkripsiCUDA(m,k,g,p,y,res);


	// printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("c[%d] = %lu 	c[%d] = %lu\n", 2*i, res[2*i], 2*i+1, res[2*i+1]);
	// }

	// printf("c ...\n");
	// printf("c[%d] = %lu 	c[%d] = %lu\n", banyakdata * 2-2, res[banyakdata * 2-2], banyakdata *2-1,res[banyakdata*2-1]);

	// printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("m[%d] = %lu\n", i, res2[i]);
	// }

	// printf("m[...]\n");
	// printf("m[%d] = %lu\n", banyakdata-1, res2[banyakdata-1]);

	free(m);
	free(k);
	free(res);

	return 0;
}
