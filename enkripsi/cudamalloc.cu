#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ul;
typedef unsigned int uint;

int banyakdata = 2560;
int dimensigrid = 20;
int dimensiblok = 128;

typedef struct {
	char size;
	uint* value;
}big;

__global__ void kernelenk(char *size, uint *value){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;
}


void CUDAenk(char *size, uint *value) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	char *devsize;
	uint *devvalue;

	cudaMalloc((void**)&devsize, banyakdata * 4 * sizeof(char));
	cudaMalloc((void**)&devvalue, banyakdata * 4 * sizeof(uint) * 12);

	cudaMemcpy(devsize, size, (sizeof(char) * banyakdata * 4), cudaMemcpyHostToDevice);
	cudaMemcpy(devvalue, value, (sizeof(uint) * 4 * banyakdata * 12), cudaMemcpyHostToDevice);


	kernelenk << <dimensigrid, dimensiblok >> >(devsize, devvalue);

	cudaDeviceSynchronize();


	// //	COPY FROM DEVICE TO HOST HERE
	// big* tempres = (big*) malloc(banyakdata * 2 * sizeof(big)); 
	// for (int i = 0; i < banyakdata*2; i++){
	// 	tempres[i].value = (uint*) malloc(sizeof(uint) * p->size);
	// }
	// cudaMemcpy(tempres, devres, (sizeof(big) * 2 * banyakdata), cudaMemcpyDeviceToHost);

	// for (int i = 0; i < banyakdata*2; i++){
	// 	res[i].size = tempres[i].size;
	// 	cudaMemcpy(res[i].value, tempres[i].value, sizeof(uint) * p->size, cudaMemcpyDeviceToHost);
	// }

	cudaFree(devsize);
	cudaFree(devvalue);
}

void mainenkripsi(char *size, uint * value){
	// printf("Encrypting...\n");
	//========================================================//

	cudaSetDevice(0);

	CUDAenk(size, value);

	cudaDeviceReset();

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Cipher %d  size %d : %u\n",i, res[i].size, res[i].value[0]);
	// }
	// printf("Cipher ... : ...\n");
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);
}

void init(char *size, uint *value){
	// Kunci publik p
	srand(2018);

	for(int i = 0 ; i < banyakdata * 4 ; i++){
		size[i] = 16;
	}

	for(int i = 0 ; i < banyakdata * 4 * 12 ; i++){
		value[i] = rand() % UINT_MAX;
	}
}

int main(){
	char *size;
	uint *value;

	size = (char*)malloc(sizeof(char) * banyakdata * 4);
	value = (uint*)malloc(sizeof(uint) * banyakdata * 4 *12);

	init(size,value);
	mainenkripsi(size,value);

	free(size);
	free(value);
	
	return 0;
}

