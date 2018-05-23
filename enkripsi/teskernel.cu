#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>


int banyakdata = 1;
int dimensigrid = 1;
int dimensiblok = 1;


__global__ void kernelenk(int *res) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int temp = 0;
	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 1000; j++)
		{
			int kali = 1+j+i;
			temp += kali;
		}
	}
	res[idx] = temp;
}

void fserial(int *res) {
	for (int kk = 0; kk < banyakdata; ++kk)
	{
		int temp = 0;
		for (int i = 0; i < 1000; i++)
		{
			for (int j = 0; j < 1000; j++)
			{
				int kali = 1+j+i;
				temp += kali;
			}
		}
		res[kk] = temp;
	}
	
}

int serial(){
	int *res;
	res = (int*) malloc(sizeof(int) * banyakdata);

	clock_t begin = clock();
		fserial(res);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi enkripsi = %f milliseconds\n", time_spent / 1000);

	for (int i = 0; i < 5; i++)
	{
		printf("Res %d : %d\n",i,res[i]);
	}
		printf("Res %d : %d\n",banyakdata-1,res[banyakdata-1]);


	free(res);

}

int paralel(){
	int *res, *devres;
	res = (int*) malloc(sizeof(int) * banyakdata);

	cudaMalloc((void**)&devres,sizeof(int) * banyakdata);
	
	kernelenk<<<dimensigrid,dimensiblok>>>(devres);
	
	cudaDeviceSynchronize();

	cudaMemcpy(res, devres, sizeof(int) * banyakdata, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; i++)
	{
		printf("Res %d : %d\n",i,res[i]);
	}
		printf("Res %d : %d\n",banyakdata-1,res[banyakdata-1]);
	
	cudaFree(devres);
	free(res);

	return 0;
}

int main(){
	// serial();
	paralel();
}