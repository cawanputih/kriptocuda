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
int sizebig = 1;

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
	//printf("res adlaah %u\n", res->value[0]);
}

__device__ void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, uint *minbuff, big *mulbuff) {
	
	// printf("y adlaah %u\n", k->value[0]);
	// BLok 1 Cipher
	modexp(g,k,p,res,minbuff,mulbuff);
	
	// Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff);

	printf("res adlaah %u\n", p->value[0]);

}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

__global__ void kernelenk(char *size, uint *value){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	int sizebig = 1;
	int banyakdata = 2560;

	__shared__ big sm[128];
	__shared__ big sk[128];
	__shared__ big smulbuff[128];
	__shared__ big sres[256];
	__shared__ big sp;
	__shared__ big sg;
	__shared__ big sy;
	__shared__ uint s[2400];

	uint *sresval = s;
	uint *smulbuffval = (uint*)&sresval[2*sizebig*128*2];
	uint *sminbuffval = (uint*)&smulbuffval[2*sizebig*128];
	uint *spval = (uint*)&sminbuffval[sizebig*128];
	uint *sgval = (uint*)&spval[sizebig];
	uint *syval = (uint*)&sgval[sizebig];
	uint *smval = (uint*)&syval[sizebig];
	uint *skval = (uint*)&smval[sizebig*128];

	sp.size = size[0];
	sg.size = size[0];
	sy.size = size[0];
	sm[jdx].size = size[0];
	sk[jdx].size = size[0];

	for (int i = 0; i < sizebig; i++)
	{
		spval[i] = value[i];
		sgval[i] = value[sizebig + i];
		syval[i] = value[2*sizebig + i];
		smval[jdx*2+i] = value[3*sizebig + idx*sizebig + i];
		skval[jdx*2+i] = value[(banyakdata + 3)*sizebig + idx*sizebig + i];
	}

	sm[jdx].value = (uint*)&smval[jdx*2];
	sk[jdx].value = (uint*)&skval[jdx*2];
	sres[2*jdx].value = (uint*)&sresval[jdx*4*2];
	sres[2*jdx+1].value = (uint*)&sresval[jdx*4*2+4];
	smulbuff[jdx].value = (uint*)&smulbuffval[jdx*4];
	sp.value = spval;
	sg.value = sgval;
	sy.value = syval;

	__syncthreads();

	enkripsi(sm + jdx, sk + jdx, &sg, &sp, &sy, sres + 2*jdx, sminbuffval+jdx, smulbuff+jdx);

	size[2*banyakdata + 3 + 2*idx] = sres[2*jdx].size;
	size[2*banyakdata + 3 + 2*idx + 1] = sres[2*jdx + 1].size;

	for (int i = 0; i < sres[2*jdx].size; i++)
	{
		value[(2*banyakdata+3)*sizebig + 2 * idx * sizebig + i] = sres[2*jdx].value[i];
	}

	for (int i = 0; i < sres[2*jdx+1].size; i++)
	{
		value[(2*banyakdata+3)*sizebig + (2 * idx + 1)* sizebig + i] = sres[2*jdx+1].value[i];
	}
}

void CUDAenk(char *size, uint *value) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	char *devsize;
	uint *devvalue;

	cudaMalloc((void**)&devsize, (banyakdata * 4 + 3 ) * sizeof(char));
	cudaMalloc((void**)&devvalue, (banyakdata * 6 + 3) * sizeof(uint) * sizebig);

	cudaMemcpy(devsize, size, (banyakdata * 4 + 3 ) * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(devvalue, value, (banyakdata * 6 + 3) * sizeof(uint) * sizebig, cudaMemcpyHostToDevice);

	kernelenk << <dimensigrid, dimensiblok >> >(devsize, devvalue);

	cudaDeviceSynchronize();


	// COPY FROM DEVICE TO HOST HERE

	cudaMemcpy(&size[3 + 2 * banyakdata], &devsize[3 + 2 *banyakdata], 2 * banyakdata, cudaMemcpyDeviceToHost);
	cudaMemcpy(&value[(3 + 2 * banyakdata) * sizebig], &devvalue[(3 + 2 * banyakdata) * sizebig], 2 * banyakdata * sizebig * 2, cudaMemcpyDeviceToHost);

	cudaFree(devsize);
	cudaFree(devvalue);
}

void init(char *size, uint *value){
	srand(2018);
	big *p, *g, *x, *y;

	p = (big*)malloc(sizeof(big));
	g = (big*)malloc(sizeof(big));
	x = (big*)malloc(sizeof(big));
	y = (big*)malloc(sizeof(big));

	p->size = sizebig;
	p->value = value;
	//p->value[0] = UINT_MAX;
	for (int i = 0; i < p->size; i++)
	{
		p->value[i] = 2357;
		//p->value[i] = rand() % UINT_MAX;
	}

	// Kunci publik g
	g->size = sizebig;
	g->value = &value[sizebig];
	for (int i = 0; i < g->size; i++)
	{
		g->value[i] = 2;
		// g->value[i] = rand() % UINT_MAX;
	}

	// Kunci privat x
	x->size = sizebig;
	x->value = (uint*) malloc(x->size * sizeof(uint));
	for (int i = 0; i < x->size; i++)
	{
		x->value[i] = 1751;
		// x->value[i] = rand() % UINT_MAX;
	}

	// Cari nilai kunci publik y = (g^x) mod p
	big* mulbuff = (big*) malloc(sizeof(big));
	mulbuff->value = (uint*) malloc(sizeof(uint) * p->size * 2);
	uint* minbuff = (uint*) malloc(sizeof(uint) * p->size * 2);

	y->value = &value[sizebig * 2];
	carikunciy(g,x,p,y,minbuff,mulbuff);

	// printf("y 0 : %u\n", y->value[0]);
	// printf("y 0 : %u\n", y->value[1]);

	//========================================================//
	// Blok plainteks dan k

	for(int i = 0 ; i < banyakdata * sizebig ; i++){
		value[3*sizebig+i] = 1001;
	}

	for(int i = 0 ; i < banyakdata * sizebig ; i++){
		value[3*sizebig+ banyakdata*sizebig + i] = 77;
	}

	for(int i = 0 ; i < (banyakdata * 4 + 3) ; i++){
		size[i] = sizebig;
	}

}

int main(){
	char *size;
	uint *value;

	size = (char*)malloc((banyakdata * 4 + 3 ) * sizeof(char));
	value = (uint*)malloc((banyakdata * 6 + 3) * sizeof(uint) * sizebig);

	init(size,value);

	// printf("Encrypting...\n");
	//========================================================//

	CUDAenk(size, value);

	for (int i = 0; i < 5; i++)
	{
		printf("Cipher %d  : %u\n",i, value[(3 + 2 * banyakdata) * sizebig + i]);
	}
	printf("Cipher ... : ...\n");
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);

	free(size);
	free(value);
	
	return 0;
}

