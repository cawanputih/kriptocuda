#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ul;
typedef unsigned int uint;


int banyakdata = 640;
int dimensigrid = 5;
int dimensiblok = 128;
int sizebig = 4;

typedef struct {
	char size;
	uint* value;
}big;


__host__ __device__ short ukuranbit(big *a) {
	uint lastval = a->value[a->size-1];
	short res = 0;
	while (lastval != 0) {
		lastval >>= 1;
		res++;
	}
	return res + (a->size - 1) * 32;
}

__host__ __device__ char getbit(big* a, short count) {
	
	return (a->value[count / 32] & ((uint) 1 << (count % 32))) != 0;
}

__host__ __device__ uint getShiftedBlock(big *num, char noblok, char geser) {
	uint part1 = (noblok == 0 || geser == 0) ? 0 : (num->value[noblok - 1] >> (32-geser));
	uint part2 = (noblok == num->size) ? 0 : (num->value[noblok] << geser);

	return part1 | part2;
}

__host__ __device__ void kali(big *a, big *b, big* res) {
	if (a->size == 0 || b->size == 0) {
		res->size = 0;
		return ;
	}
	char ukurana = a->size;
	char ukuranb = b->size;
	char ukuranres = ukurana + ukuranb;
	res->size = ukuranres;

	for (char i = 0; i < ukuranres; i++) {
		res->value[i] = 0;
	}

	for (char i = 0; i < ukurana; i++) {
			uint aval = a->value[i];

			if (aval==0){
				continue;
			}
			uint lebih = 0;
			for (char j = 0, lebih = 0; j < ukuranb; j++) {
				uint bval = b->value[j];
				ul temp = res->value[i+j] + aval * bval + lebih;
				res->value[i+j] = temp % UINT_MAX;
				lebih = temp / UINT_MAX;
			}
			res->value[i+ukuranb] = lebih;
	}

	if (res->value[res->size - 1] == 0){
		res->size--;
	}
}

__host__ __device__ void modulo(big* a, big* b, big* res, uint* minbuff) {
	res->size = a->size;
	for(char i = 0 ; i < res->size ;i++){
		res->value[i] = a->value[i];
	}

	if (a->size < b->size) {
		return ;
	}

	char i, j, k;
	char i2;
	uint temp ;
	char borrowIn, borrowOut;

	char ukurana = a->size;
	char ukuranb = b->size;

	res->value[res->size] = 0;
	res->size++;

	i = ukurana - ukuranb + 1;
	while (i > 0) {
		i--;
		i2 = 32;
		while (i2 > 0) {
			i2--;
			for (j = 0, k = i, borrowIn = 0; j <= ukuranb; j++, k++) {
				temp = res->value[k] - getShiftedBlock(b, j, i2);
				borrowOut = (temp > res->value[k]);
				if (borrowIn) {
					borrowOut |= (temp == 0);
					temp--;
				}
				minbuff[k] = temp; 
				borrowIn = borrowOut;
			}

			for (; k < ukurana && borrowIn; k++) {
				borrowIn = (res->value[k] == 0);
				minbuff[k] = res->value[k] - 1;
			}

			if (!borrowIn) {
				while (k > i) {
					k--;
					res->value[k] = minbuff[k];
				}
			} 
		}
	}

	while (res->size > 0 && res->value[res->size - 1] == 0)
		res->size--;
}

void tambah(big* a, char b, big* res) {
	if (a->size == 0) {
		res->size = 1;
		res->value[0] = uint(b);
		return;
	}

	char carryIn = 0;
	uint temp;

	res->size = a->size + 1;

	res->value[0] = a->value[0] + (uint)b;
	carryIn = (res->value[0] < a->value[0]);
	char i = 1;
	for (; i < a->size && carryIn; i++) {
		temp = a->value[i] + (uint)1;
		carryIn = (temp == 0);
		res->value[i] = temp;
	}

	for (; i < a->size; i++)
		res->value[i] = a->value[i];

	if (carryIn)
		res->value[i] = 1;
	else
		res->size--;
}

void kurang(big* a, big *b, big* res) {
	res->size = a->size;
	for (int i = 0; i < res->size; i++){
		res->value[i] = 0;
	}

	if (b->size == 0) {
		return;
	} 

	char borrowIn, borrowOut;
	uint temp;
	char i;

	for (i = 0, borrowIn = 0; i < b->size; i++) {
		temp = a->value[i] - b->value[i];
		borrowOut = (temp > a->value[i]);
		if (borrowIn) {
			borrowOut |= (temp == 0);
			temp--;
		}
		res->value[i] = temp;
		borrowIn = borrowOut;
	}

	for (; i < a->size && borrowIn; i++) {
		borrowIn = (a->value[i] == 0);
		res->value[i] = a->value[i] - 1;
	}

	for (; i < a->size; i++)
		res->value[i] = a->value[i];

	if (res->value[res->size - 1] == 0){
		res->size--;
	}
}

__host__ __device__ void modexp(big* a, big* b, big* c, big* res, uint* minbuff, big* mulbuff){
	//printf("c val 0 %u\n", c->value[0]);
	res->size = 1;
	res->value[0] = 1;

	short i = ukuranbit(b);
	while (i > 0) {
		i--;
		kali(res,res,mulbuff);
		modulo(mulbuff,c,res,minbuff);
		if (getbit(b,i)) {
			kali(res, a, mulbuff);
			modulo(mulbuff, c, res, minbuff);
		}
	}

}

__device__ void dekripsi(big *c1, big *c2, big *p, big *e, big *res, uint *minbuff, big *mulbuff) {
	modexp(c1,e,p,res,minbuff,mulbuff);
	kali(res, c2, mulbuff);
	modulo(mulbuff, p, res, minbuff);
	// printf("c1 adlaah %u\n", c1->value[0]);
	// printf("c2 adlaah %u\n", c2->value[0]);
}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

__global__ void kerneldek(uint *p, uint *e, uint *c, uint *resval, char *ressize, uint *buffmin, uint *buffmul){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	int sizebig = 4;
	// int banyakdata = 256;

	__shared__ big sa[128];
	__shared__ big sb[128];
	__shared__ big smulbuff[128];
	__shared__ big sres[256];
	__shared__ big sp;
	__shared__ big se;
	__shared__ uint s[3200];

	uint *spval = s;
	uint *seval = (uint*)&spval[sizebig];
	uint *sresval = (uint*)&seval[sizebig];
	uint *smulbuffval = (uint*)&sresval[sizebig*128*2];
	//uint *sminbuffval = (uint*)&smulbuffval[2*sizebig*128];
	//uint *sminbuffval = (uint*)&sresval[2*sizebig*128*2];
	uint *saval = (uint*)&smulbuffval[sizebig*128*2];
	uint *sbval = (uint*)&saval[sizebig*128];

	for (int i = 0; i < sizebig; i++)
	{
		spval[i] = p[i];
		seval[i] = e[i];
		saval[jdx*sizebig+i] = c[2*idx*sizebig+i];
		sbval[jdx*sizebig+i] = c[(2*idx+1)*sizebig+i];
	}

	sp.size = sizebig;
	se.size = sizebig;
	
	sa[jdx].size = sizebig;
	sb[jdx].size = sizebig;

	sp.value = spval;
	se.value = seval;

	sa[jdx].value = (uint*)&saval[jdx*sizebig];
	sb[jdx].value = (uint*)&sbval[jdx*sizebig];

	sres[jdx].value = (uint*)&sresval[jdx*sizebig*2];
	smulbuff[jdx].value = (uint*)&smulbuffval[jdx*sizebig*2];
	// sminbuff[jdx].value = (uint*)&sminbuffval[jdx*sizebig];

	__syncthreads();

	//uint* minbuff = (uint*) malloc(sizeof(uint) * sizebig);

	dekripsi(sa + jdx, sb + jdx, &sp, &se, sres + jdx, buffmin + 2 *sizebig * idx, smulbuff + jdx);

	ressize[idx] = sres[jdx].size;
	
	for (int i = 0; i < sres[jdx].size; i++)
	{
		resval[idx * sizebig * 2 + i] = sres[jdx].value[i];
	}
}

void CUDAenk(uint *p, uint *e, uint *c, uint *resval, char *ressize) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	char *devressize;
	uint *devp, *deve, *devc, *devresval, *buffmin, *buffmul;

	cudaMalloc((void**)&devp, sizebig * sizeof(uint));
	cudaMalloc((void**)&deve, sizebig * sizeof(uint));

	cudaMalloc((void**)&devc, 2 * banyakdata * sizebig * sizeof(uint));

	cudaMalloc((void**)&devresval, banyakdata * 2 * sizebig * sizeof(uint));
	cudaMalloc((void**)&devressize,  banyakdata * sizeof(char));

	cudaMalloc((void**)&buffmin, banyakdata * sizebig * 2 * sizeof(uint));
	cudaMalloc((void**)&buffmul, banyakdata * sizebig * 2 * sizeof(uint));

	cudaMemcpy(devp, p, sizebig * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(deve, e, sizebig * sizeof(uint), cudaMemcpyHostToDevice);

	cudaMemcpy(devc, c, 2 * banyakdata * sizebig * sizeof(uint), cudaMemcpyHostToDevice);

		kerneldek << <dimensigrid, dimensiblok >> >(devp, deve, devc, devresval, devressize, buffmin, buffmul);

	cudaDeviceSynchronize();

	// COPY FROM DEVICE TO HOST HERE

	cudaMemcpy(ressize, devressize, banyakdata, cudaMemcpyDeviceToHost);
	cudaMemcpy(resval, devresval, banyakdata * 2 * sizebig * sizeof(uint), cudaMemcpyDeviceToHost);

	cudaFree(devp);
	cudaFree(deve);
	cudaFree(devc);
	cudaFree(devresval);
	cudaFree(devressize);
	cudaFree(buffmin);
	cudaFree(buffmul);
}

void init(uint *pval, uint *eval, uint *cval){
	srand(2018);
	big *p, *e;

	p = (big*)malloc(sizeof(big));
	e = (big*)malloc(sizeof(big));
	
	p->size = sizebig;
	p->value = pval;
	p->value[0] = UINT_MAX;
	for (int i = 0; i < p->size; i++)
	{
		//p->value[i] = 2357;
		p->value[i] = rand() % UINT_MAX;
	}

	e->size = sizebig;
	e->value = eval;
	for (int i = 0; i < e->size; i++)
	{
		// e->value[i] = 2;
		e->value[i] = rand() % UINT_MAX;
	}


	//========================================================//
	// Blok cipherteks

	for(int i = 0 ; i < 2 * banyakdata * sizebig ; i++){
		cval[i] = rand() % UINT_MAX;
	}
}

int main(){
	char *ressize;
	uint *p, *e, *c, *resval;

	p = (uint*) malloc(sizebig * sizeof(uint));
	e = (uint*) malloc(sizebig * sizeof(uint));
	
	c = (uint*) malloc(2 * banyakdata * sizebig * sizeof(uint));

	resval = (uint*) malloc(banyakdata * 2 * sizebig * sizeof(uint));
	ressize = (char*) malloc(banyakdata * sizeof(char));

	init(p,e,c);

	// printf("Encrypting...\n");
	//========================================================//

	CUDAenk(p,e,c,resval,ressize);

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Plain %d  size %d : %u\n",i, ressize[i], resval[i*2*sizebig]);
	// }
	// printf("Plain ... : ...\n");
	// printf("Plain %d  size %d : %u\n",banyakdata-2, ressize[banyakdata-2], resval[(banyakdata-2) * 2 * sizebig]);
	// printf("Plain %d  size %d : %u\n",banyakdata-1, ressize[banyakdata-1], resval[(banyakdata-1) * 2 * sizebig]);


	free(p);
	free(e);
	free(c);
	free(resval);
	free(ressize);
	
	return 0;
}

