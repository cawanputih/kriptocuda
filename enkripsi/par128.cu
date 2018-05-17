#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ul;
typedef unsigned int uint;


int banyakdata = 1280;
int dimensigrid = 10;
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

__device__ void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, uint *minbuff, big *mulbuff) {
	

	//printf("res adlaah tanga\n");
	// BLok 1 Cipher
	modexp(g,k,p,res,minbuff,mulbuff);
	//printf("res 0 val 0 %u\n", res->value[0]);
	
	// Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff);
	//printf("res 1 val 0 %u\n", (res+1)->value[0]);

}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

__global__ void kernelenk(uint *p, uint *g, uint *y, uint *m, uint *k, uint *resval, char *ressize, uint *buffmin, uint *buffmul){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	int sizebig = 4;
	// int banyakdata = 256;

	__shared__ big sm[128];
	__shared__ big sk[128];
	__shared__ big smulbuff[128];
	__shared__ big sres[256];
	__shared__ big sp;
	__shared__ big sg;
	__shared__ big sy;
	__shared__ uint s[4200];

	uint *spval = s;
	uint *sgval = (uint*)&spval[sizebig];
	uint *syval = (uint*)&sgval[sizebig];
	uint *sresval = (uint*)&syval[sizebig];
	uint *smulbuffval = (uint*)&sresval[2*sizebig*128*2];
	//uint *sminbuffval = (uint*)&smulbuffval[2*sizebig*128];
	//uint *sminbuffval = (uint*)&sresval[2*sizebig*128*2];
	uint *smval = (uint*)&smulbuffval[2*sizebig*128];
	uint *skval = (uint*)&smval[sizebig*128];

	for (int i = 0; i < sizebig; i++)
	{
		spval[i] = p[i];
		sgval[i] = g[i];
		syval[i] = y[i];
		smval[jdx*sizebig+i] = m[idx*sizebig + i];
		skval[jdx*sizebig+i] = k[idx*sizebig + i];
	}

	sp.size = sizebig;
	sg.size = sizebig;
	sy.size = sizebig;
	sm[jdx].size = sizebig;
	sk[jdx].size = sizebig;

	sp.value = spval;
	sg.value = sgval;
	sy.value = syval;
	sm[jdx].value = (uint*)&smval[jdx*sizebig];
	sk[jdx].value = (uint*)&skval[jdx*sizebig];
	sres[2*jdx].value = (uint*)&sresval[jdx*sizebig*4];
	sres[2*jdx+1].value = (uint*)&sresval[jdx*sizebig*4+sizebig*2];
	smulbuff[jdx].value = (uint*)&smulbuffval[jdx*sizebig*2];
	// sminbuff[jdx].value = (uint*)&sminbuffval[jdx*sizebig];

	__syncthreads();

	//uint* minbuff = (uint*) malloc(sizeof(uint) * sizebig);

	enkripsi(sm + jdx, sk + jdx, &sg, &sp, &sy, sres + 2*jdx, buffmin + 2 *sizebig * idx, smulbuff + jdx);

	ressize[2*idx] = sres[2*jdx].size;
	ressize[2*idx + 1] = sres[2*jdx + 1].size;

	for (int i = 0; i < sres[2*jdx].size; i++)
	{
		resval[2 * idx * sizebig * 2 + i] = sres[2*jdx].value[i];
	}

	for (int i = 0; i < sres[2*jdx+1].size; i++)
	{
		resval[(2 * idx + 1)* sizebig * 2 + i] = sres[2*jdx+1].value[i];
	}
}

void CUDAenk(uint *p, uint *g, uint *y, uint *m, uint *k, uint *resval, char *ressize) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	char *devressize;
	uint *devp, *devg, *devy, *devm, *devk, *devresval, *buffmin, *buffmul;

	cudaMalloc((void**)&devp, sizebig * sizeof(uint));
	cudaMalloc((void**)&devg, sizebig * sizeof(uint));
	cudaMalloc((void**)&devy, sizebig * sizeof(uint));

	cudaMalloc((void**)&devm, banyakdata * sizebig * sizeof(uint));
	cudaMalloc((void**)&devk, banyakdata * sizebig * sizeof(uint));
	cudaMalloc((void**)&devresval, 2 * banyakdata * 2 * sizebig * sizeof(uint));
	
	cudaMalloc((void**)&devressize, 2 * banyakdata * sizeof(char));

	cudaMalloc((void**)&buffmin, banyakdata * sizebig * 2 * sizeof(uint));
	cudaMalloc((void**)&buffmul, banyakdata * sizebig * 2 * sizeof(uint));

	cudaMemcpy(devp, p, sizebig * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(devg, g, sizebig * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(devy, y, sizebig * sizeof(uint), cudaMemcpyHostToDevice);
	
	cudaMemcpy(devm, m, banyakdata * sizebig * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(devk, k, banyakdata * sizebig * sizeof(uint), cudaMemcpyHostToDevice);

		kernelenk << <dimensigrid, dimensiblok >> >(devp, devg, devy, devm, devk, devresval, devressize, buffmin, buffmul);

	cudaDeviceSynchronize();

	// COPY FROM DEVICE TO HOST HERE

	cudaMemcpy(ressize, devressize, 2 * banyakdata, cudaMemcpyDeviceToHost);
	cudaMemcpy(resval, devresval, 2 * banyakdata * 2 * sizebig * sizeof(uint), cudaMemcpyDeviceToHost);

	cudaFree(devp);
	cudaFree(devg);
	cudaFree(devy);
	cudaFree(devm);
	cudaFree(devk);
	cudaFree(devresval);
	cudaFree(devressize);
	cudaFree(buffmin);
	cudaFree(buffmul);
}

void init(uint *pval, uint *gval, uint *yval, uint *mval, uint *kval){
	srand(2018);
	big *p, *g, *x, *y;

	p = (big*)malloc(sizeof(big));
	g = (big*)malloc(sizeof(big));
	x = (big*)malloc(sizeof(big));
	y = (big*)malloc(sizeof(big));

	p->size = sizebig;
	p->value = pval;
	p->value[0] = UINT_MAX;
	for (int i = 0; i < p->size; i++)
	{
		//p->value[i] = 2357;
		p->value[i] = rand() % UINT_MAX;
	}

	// Kunci publik g
	g->size = sizebig;
	g->value = gval;
	for (int i = 0; i < g->size; i++)
	{
		// g->value[i] = 2;
		g->value[i] = rand() % UINT_MAX;
	}

	// Kunci privat x
	x->size = sizebig;
	x->value = (uint*) malloc(x->size * sizeof(uint));
	for (int i = 0; i < x->size; i++)
	{
		// x->value[i] = 1751;
		x->value[i] = rand() % UINT_MAX;
	}

	// Cari nilai kunci publik y = (g^x) mod p
	big* mulbuff = (big*) malloc(sizeof(big));
	mulbuff->value = (uint*) malloc(sizeof(uint) * sizebig * 2);
	uint* minbuff = (uint*) malloc(sizeof(uint) * sizebig * 2);

	y->value = (uint*) malloc(sizeof(uint) * sizebig * 2);
	carikunciy(g,x,p,y,minbuff,mulbuff);

	for (int i = 0; i < sizebig; i++)
	{
		yval[i] = y->value[i];
	}

	// printf("y size %d : %u\n", y->size, y->value[0]);

	//========================================================//
	// Blok plainteks dan k

	for(int i = 0 ; i < banyakdata * sizebig ; i++){
		// mval[i] = 1001;
		mval[i] = rand() % UINT_MAX;
		// kval[i] = 77;
		kval[i] = rand() % UINT_MAX;
	}
}

int main(){
	char *ressize;
	uint *p, *g, *y, *m, *k, *resval;

	p = (uint*) malloc(sizebig * sizeof(uint));
	g = (uint*) malloc(sizebig * sizeof(uint));
	y = (uint*) malloc(sizebig * sizeof(uint));

	m = (uint*) malloc(banyakdata * sizebig * sizeof(uint));
	k = (uint*) malloc(banyakdata * sizebig * sizeof(uint));
	resval = (uint*) malloc(2 * banyakdata * 2 * sizebig * sizeof(uint));
	ressize = (char*) malloc(2 * banyakdata * sizeof(char));

	init(p,g,y,m,k);

	// printf("Encrypting...\n");
	//========================================================//

	CUDAenk(p,g,y,m,k,resval,ressize);

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Cipher %d  size %d : %u\n",i, ressize[i], resval[i*2*sizebig]);
	// }
	// printf("Cipher ... : ...\n");
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-2, ressize[banyakdata*2-2], resval[(banyakdata*2-2) * 2 * sizebig]);
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-1, ressize[banyakdata*2-1], resval[(banyakdata*2-1) * 2 * sizebig]);


	free(p);
	free(g);
	free(y);
	free(m);
	free(k);
	free(resval);
	free(ressize);
	
	return 0;
}

