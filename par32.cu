#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 10240;
int dimensigrid = 80;
int dimensiblok = 128;

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

cudaError_t enkripsiCUDA(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	cudaError_t cudaStatus;

	cudaSetDevice(0);

	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devm, *devk, *devres;
	


	cudaMalloc((void**)&devm, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devk, banyakdata * sizeof(ulint));
	cudaMalloc((void**)&devres, banyakdata * 2 * sizeof(ulint));
	
	cudaMemcpy((devm), m, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);
	cudaMemcpy((devk), k, (sizeof(ulint) * banyakdata), cudaMemcpyHostToDevice);	

		// size_t free_byte ;
  //       size_t total_byte ;
  //       cudaMemGetInfo( &free_byte, &total_byte ) ;
		// double free_db = (double)free_byte ;
  //       double total_db = (double)total_byte ;
  //       double used_db = total_db - free_db ;

  //       printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

  //           used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

	// printf("<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>>>\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernelenk << <dimensigrid, dimensiblok>> >(devm,devk,g,p,y,devres);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nDurasi enkripsi= %f ms\n", milliseconds);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else {
		//printf("Success\n");
	}

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res, devres, (sizeof(ulint) * 2 * banyakdata), cudaMemcpyDeviceToHost);


	Error:

	
	cudaFree(devm);
	cudaFree(devk);
	cudaFree(devres);

	return cudaStatus;
}

cudaError_t dekripsiCUDA(ulint *c, ulint p, ulint e, ulint *res2) {
	cudaError_t cudaStatus;

	cudaSetDevice(0);

	//=====================BAGIAN M[] K[] DAN RES[] ====================================//
	ulint *devc, *devres2;
	

	cudaMalloc((void**)&devc, banyakdata * 2 * sizeof(ulint));
	cudaMalloc((void**)&devres2, banyakdata * sizeof(ulint));
	
	cudaMemcpy((devc), c, (sizeof(ulint) * banyakdata * 2), cudaMemcpyHostToDevice);

	// printf("<<<<<<<<<<<<<<<<<<KERNEL>>>>>>>>>>>>>>>>>\n");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kerneldek << <dimensigrid, dimensiblok>> >(devc,p,e,devres2);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nDurasi enkripsi= %f ms\n", milliseconds);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else {
		//printf("Success\n");
	}

	cudaDeviceSynchronize();

	//	COPY FROM DEVICE TO HOST HERE 
	cudaMemcpy(res2, devres2, (sizeof(ulint) * banyakdata), cudaMemcpyDeviceToHost);


	Error:

	
	cudaFree(devc);
	cudaFree(devres2);

	return cudaStatus;

}

void initenkripsi(ulint *m, ulint *k) {
	for (int i = 0; i < banyakdata; i++)
    {
        m[i] = 0;
    }

	FILE *file = fopen("plain.plain", "r");
    char *code;
    size_t n = 0;
    int c;

    code = (char*) malloc(9999999);
    
    while ((c = fgetc(file)) != EOF)
    {
        code[n++] = (char) c;
    }
    code[n] = '\0';

 
    char karakter = code[0];

    int i = 0;
    int indexpesan = -1;
    while(karakter != '\0'){
        karakter = code[i];
        if(i % 3== 0){
            indexpesan++;
            m[indexpesan] += karakter * 1000000;
        }else if(i % 3 ==1){
            m[indexpesan] += karakter * 1000;
        }else{
            m[indexpesan] += karakter;
        }
        i++;
    }

    // printf("count : %d\n", indexpesan);

    // nilai k //
	srand(2018);

	for (int i = 0; i < banyakdata; i++) {
		k[i] = rand() % 3999999978;
	}
}

ulint stringtolong(char* s){
	ulint res = 0;
	int i = 0;
	while(s[i] != '\0'){
		res *= 10;
		res += s[i] - '0';
		i++;
	}
	return res;
}

void initdekripsi(ulint *c) {
	for (int i = 0; i < banyakdata*2; i++)
	{
		c[i] = 0;
	}

	char *buffer = 0;
	long length;
	FILE *f = fopen("cipher.cipher", "rb");

	if (f)
	{
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		fseek(f, 0, SEEK_SET);
		buffer = (char*)malloc(length);
		if (buffer) {
			fread(buffer, 1, length, f);
		}
		buffer[length] = '\0';
		fclose(f);
	}
	char delimstrip[2];
	delimstrip[0] = 45;
	delimstrip[1] = 0;

	// Baca seluruh ciphertext
	char *tempsplit;
	tempsplit = strdup(strtok(buffer, delimstrip));
	c[0] = stringtolong(tempsplit);
	tempsplit = strdup(strtok(NULL, delimstrip));
	c[1] = stringtolong(tempsplit);
	// Baca m
	for (int i = 1; i < banyakdata; i++) {
		tempsplit = strdup(strtok(NULL, delimstrip));
		c[2*i] = stringtolong(tempsplit);
		tempsplit = strdup(strtok(NULL, delimstrip));
		c[2*i+1] = stringtolong(tempsplit);
	}
}

void initenkripsi2(ulint *m, ulint *k){
	
	for (int i = 0; i < banyakdata; i++) {
		m[i] = rand() % 3999999978;
		k[i] = rand() % 3999999978;
	}	
}

void writecipher(ulint* c){
	FILE *fp = fopen("cipher.cipher","w");
	

	for (int i = 0; i < banyakdata*2; i++)
	{
		fprintf(fp, "%lu", c[i]);
		fprintf(fp, "%c", '-');
	}

	fclose(fp);
}

void writedekrip(ulint* m){
	FILE *fp = fopen("dekrip.dekrip","w");
	

	for (int i = 0; i < banyakdata; i++)
	{
		ulint temp = m[i];
		fprintf(fp, "%c",  temp/1000000 );
		fprintf(fp, "%c",  (temp/1000) % 1000 );
		fprintf(fp, "%c",  temp % 1000);
	}

	fclose(fp);
}

int main(){
	ulint *m, *k, *res, *res2, g, p, y, x, e, *res3;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	k = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * 2 * sizeof(ulint));
	res2 = (ulint*)malloc(banyakdata * sizeof(ulint));
	res3 = (ulint*)malloc(banyakdata * 2 *sizeof(ulint));

	srand(2018);

	g = rand() % 3999999978;
	p = 3999999979;
	x = rand() % 3999999978;
	modexp(g,x,p,&y);
	initenkripsi(m, k);
	// initenkripsi2(m, k);


	printf("<<<<<<<<<<<<<<Pesan Asli>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %lu\n", i, m[i]);
	}

	printf("m[...]\n");
	printf("m[%d] = %lu\n", banyakdata-1, m[banyakdata-1]);

	enkripsiCUDA(m,k,g,p,y,res);

	printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("c[%d] = %lu 	c[%d] = %lu\n", 2*i, res[2*i], 2*i+1, res[2*i+1]);
	}

	printf("c ...\n");
	printf("c[%d] = %lu 	c[%d] = %lu\n", banyakdata * 2-2, res[banyakdata * 2-2], banyakdata *2-1,res[banyakdata*2-1]);

	writecipher(res);

	initdekripsi(res3);

	e = p-x-1;
	dekripsiCUDA(res3,p,e,res2);

	printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	for (int i = 0; i < 4; i++) {
		printf("m[%d] = %lu\n", i, res2[i]);
	}

	printf("m[...]\n");
	printf("m[%d] = %lu\n", banyakdata-1, res2[banyakdata-1]);
	writedekrip(res2);

	free(m);
	free(k);
	free(res);
	free(res2);

	return 0;
}