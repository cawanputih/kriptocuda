#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ul;
typedef unsigned int uint;

int banyakdata = 256000;
int dimensigrid = 2000;
int dimensiblok = 128;

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
	// printf("res adlaah %u\n", res->value[0]);
}

__device__ void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff) {
	// BLok 1 Cipher
	modexp(g,k,p,res,minbuff->value,mulbuff);
	
	// Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff->value,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff->value);
}

__global__ void kernelenk(big *m, big *k, big *g, big *p, big *y, big *res){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	__shared__ big sm[128];
	__shared__ big sk[128];
	__shared__ big sres[256];
	__shared__ big sp;
	__shared__ big sg;
	__shared__ big sy;
	__shared__ uint s[1600];

	uint *sresval = s;
	uint *spval = (uint*)&sresval[2*128*2];
	uint *sgval = (uint*)&spval[1];
	uint *syval = (uint*)&sgval[1];
	uint *smval = (uint*)&syval[1];
	uint *skval = (uint*)&smval[1*128];


	sm[jdx].size = m[idx].size;
	sk[jdx].size = k[idx].size;
	sp.size = p[0].size;
	sg.size = g[0].size;
	sy.size = y[0].size;


		smval[jdx] = m[idx].value[0];
		skval[jdx] = k[idx].value[0];
		spval[0] = p[0].value[0];
		sgval[0] = g[0].value[0];
		syval[0] = y[0].value[0];

	sm[jdx].value = (uint*)&smval[jdx];
	sk[jdx].value = (uint*)&skval[jdx];
	sres[2*jdx].value = (uint*)&sresval[jdx*2*2];
	sres[2*jdx+1].value = (uint*)&sresval[jdx*2*2+2];
	sp.value = spval;
	sg.value = sgval;
	sy.value = syval;

	__syncthreads();


	big* minbuff = (big*) malloc(sizeof(big));
	big* mulbuff = (big*) malloc(sizeof(big));

	minbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);
	mulbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);

	enkripsi(sm + jdx, sk + jdx, &sg, &sp, &sy, sres + 2*jdx, minbuff, mulbuff);

	res[2*idx].size = sres[2*jdx].size;
	res[2*idx+1].size = sres[2*jdx+1].size;

	for (int i = 0; i < sres[2*jdx].size; i++)
	{
		res[2*idx].value[i] = sres[2*jdx].value[i];
	}

	for (int i = 0; i < sres[2*jdx+1].size; i++)
	{
		res[2*idx+1].value[i] = sres[2*jdx+1].value[i];
	}
}


void CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	big *devg, *devp, *devy;

	cudaMalloc((void**)&devg, sizeof(big));
	cudaMalloc((void**)&devp, sizeof(big));
	cudaMalloc((void**)&devy, sizeof(big));

	uint *darrg, *darrp, *darry;
	cudaMalloc((void**)&darrg, g->size * sizeof(uint));
	cudaMalloc((void**)&darrp, p->size * sizeof(uint));
	cudaMalloc((void**)&darry, y->size * sizeof(uint));

	big tempg;
	cudaMemcpy(darrg, g->value, (sizeof(uint) * g->size), cudaMemcpyHostToDevice);
	tempg.size = g->size;
	tempg.value = darrg;
	cudaMemcpy((devg), &tempg, (sizeof(big)), cudaMemcpyHostToDevice);

	big tempp;
	cudaMemcpy(darrp, p->value, (sizeof(uint) * p->size), cudaMemcpyHostToDevice);
	tempp.size = p->size;
	tempp.value = darrp;
	cudaMemcpy((devp), &tempp, (sizeof(big)), cudaMemcpyHostToDevice);

	big tempy;
	cudaMemcpy(darry, y->value, (sizeof(uint) * y->size), cudaMemcpyHostToDevice);
	tempy.size = y->size;
	tempy.value = darry;
	cudaMemcpy((devy), &tempy, (sizeof(big)), cudaMemcpyHostToDevice);
	//=====================BAGIAN M[] DAN K[] ====================================//
	big *devm, *devk, *devres, *minbuff, *mulbuff;

	cudaMalloc((void**)&devm, banyakdata * sizeof(big));
	cudaMalloc((void**)&devk, banyakdata * sizeof(big));
	cudaMalloc((void**)&devres, banyakdata  * 2 *sizeof(big));
	cudaMalloc((void**)&minbuff, banyakdata  * sizeof(big));
	cudaMalloc((void**)&mulbuff, banyakdata  * sizeof(big));

	uint **tempvalue = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue2 = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue3a = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue3b = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue4 = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue5 = (uint**)malloc(sizeof(uint*)*banyakdata);

	// Alokasi Memori untuk blok m dan k
	for (int i = 0; i < banyakdata; i++) {
		big temp;
		cudaMalloc((void**)&tempvalue[i], (sizeof(uint) * p->size));
		cudaMemcpy(tempvalue[i], m[i].value, (sizeof(uint) * m[i].size), cudaMemcpyHostToDevice);
		temp.size = m[i].size;
		temp.value = tempvalue[i];
		cudaMemcpy((devm + i), &temp, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp2;
		cudaMalloc((void**)&tempvalue2[i], (sizeof(uint) * p->size));
		cudaMemcpy(tempvalue2[i], k[i].value, (sizeof(uint) * k[i].size), cudaMemcpyHostToDevice);
		temp2.size = k[i].size;
		temp2.value = tempvalue2[i];
		cudaMemcpy((devk + i), &temp2, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp3a;
		cudaMalloc((void**)&tempvalue3a[i], (sizeof(uint) * p->size * 2));
		temp3a.value = tempvalue3a[i];
		cudaMemcpy((devres + 2 * i), &temp3a, (sizeof(big)), cudaMemcpyHostToDevice);
		
		big temp3b;
		cudaMalloc((void**)&tempvalue3b[i], (sizeof(uint) * p->size * 2));
		temp3b.value = tempvalue3b[i];
		cudaMemcpy((devres + 2 * i + 1), &temp3b, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp4;
		cudaMalloc((void**)&tempvalue4[i], (sizeof(uint) * p->size * 2));
		temp4.value = tempvalue4[i];
		cudaMemcpy((minbuff + i), &temp4, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp5;
		cudaMalloc((void**)&tempvalue5[i], (sizeof(uint) * p->size * 2));
		temp5.value = tempvalue5[i];
		cudaMemcpy((mulbuff + i), &temp5, (sizeof(big)), cudaMemcpyHostToDevice);
	}

	// size_t free_byte ;
 //    size_t total_byte ;
 //    cudaMemGetInfo( &free_byte, &total_byte ) ;
	// double free_db = (double)free_byte ;
 //    double total_db = (double)total_byte ;
 //    double used_db = total_db - free_db ;


	kernelenk << <dimensigrid, dimensiblok >> >(devm, devk, devg, devp, devy, devres);

	cudaDeviceSynchronize();


	//	COPY FROM DEVICE TO HOST HERE
	big* tempres = (big*) malloc(banyakdata * 2 * sizeof(big)); 
	for (int i = 0; i < banyakdata*2; i++){
		tempres[i].value = (uint*) malloc(sizeof(uint) * p->size);
	}
	cudaMemcpy(tempres, devres, (sizeof(big) * 2 * banyakdata), cudaMemcpyDeviceToHost);

	for (int i = 0; i < banyakdata*2; i++){
		res[i].size = tempres[i].size;
		cudaMemcpy(res[i].value, tempres[i].value, sizeof(uint) * p->size, cudaMemcpyDeviceToHost);
	}

	cudaFree(darrg);
	cudaFree(darrp);
	cudaFree(darry);
	cudaFree(devg);
	cudaFree(devp);
	cudaFree(devy);

	for (int i = 0; i < banyakdata; i++) {
		cudaFree(tempvalue[i]);
		cudaFree(tempvalue2[i]);
		cudaFree(tempvalue3a[i]);
		cudaFree(tempvalue3b[i]);
		cudaFree(tempvalue4[i]);
		cudaFree(tempvalue5[i]);
	}

	free(tempvalue);
	free(tempvalue2);
	free(tempvalue3a);
	free(tempvalue3b);
	free(tempvalue4);
	free(tempvalue5);
	cudaFree(devm);
	cudaFree(devk);
	cudaFree(devres);
	cudaFree(minbuff);
	cudaFree(mulbuff);

	free(tempres);

	//cudaProfilerStop();
	//free(med);
}

void mainenkripsi(big *m, big *k, big *res, big *g, big *p, big *y){
	// printf("Encrypting...\n");
	//========================================================//

	cudaSetDevice(0);

	CUDAenk(m, k, g, p, y, res);

	cudaDeviceReset();

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Cipher %d  size %d : %u\n",i, res[i].size, res[i].value[0]);
	// }
	// printf("Cipher ... : ...\n");
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);
}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

void init(big *p, big *g, big *x, big*e, big *y, big *m, big *k, big *res){
	// Kunci publik p
	srand(2018);

	p->size = 1;
	p->value = (uint*) malloc(p->size * sizeof(uint));
	p->value[0] = UINT_MAX;
	for (int i = 1; i < p->size; i++)
	{
		p->value[i] = rand() % UINT_MAX;
	}


	// Kunci publik g
	g->size = 1;
	g->value = (uint*) malloc(g->size * sizeof(uint));
	for (int i = 0; i < g->size; i++)
	{
		// g->value[i] = 2;
		g->value[i] = rand() % UINT_MAX;
	}

	// Kunci privat x
	x->size = 1;
	x->value = (uint*) malloc(x->size * sizeof(uint));
	for (int i = 0; i < x->size; i++)
	{
		// x->value[i] = 1751;
		x->value[i] = rand() % UINT_MAX;
	}

	// Cari nilai eksponen e = (p-x-1) untuk dekripsi
	big *xplus1 = (big*) malloc(sizeof(big));
	xplus1->value = (uint*) malloc(p->size * sizeof(uint));
	e->value = (uint*) malloc(p->size * sizeof(uint));

	tambah(x, 1, xplus1);
	kurang(p,xplus1,e);

	free(xplus1->value);
	free(xplus1);

	// Cari nilai kunci publik y = (g^x) mod p
	big* mulbuff = (big*) malloc(sizeof(big));
	mulbuff->value = (uint*) malloc(sizeof(uint) * p->size * 2);
	uint* minbuff = (uint*) malloc(sizeof(uint) * p->size * 2);

	y->value = (uint*) malloc(p->size * 2 * sizeof(uint));
	carikunciy(g,x,p,y,minbuff,mulbuff);

	// printf("y 0 : %u\n", y->value[0]);
	// printf("y 0 : %u\n", y->value[1]);

	//========================================================//
	// Blok plainteks
	for(int i = 0 ; i < banyakdata ; i++){
		m[i].size = 1;
		m[i].value = (uint*) malloc(m[i].size * sizeof(uint));
		for (int j = 0; j < m[i].size; j++)
		{
			m[i].value[j] = rand() % UINT_MAX;
		}

		// Nilai k masing-masing blok
		k[i].size = 1;
		k[i].value = (uint*) malloc(k[i].size * sizeof(uint));
		for (int j = 0; j < k[i].size; j++)
		{
			// k[i].value[j] = 77;
			k[i].value[j] = rand() % UINT_MAX;
		}
	}

	// Alokasi memori untuk result
	for (int i = 0; i < banyakdata*2; i++)
	{
		res[i].value = (uint*) malloc(sizeof(uint) * p->size);
	}
}

int main(){
	big *p, *g, *x, *e, *y, *m, *k, *res;
	p = (big*)malloc(sizeof(big));
	g = (big*)malloc(sizeof(big));
	x = (big*)malloc(sizeof(big));
	e = (big*)malloc(sizeof(big));
	y = (big*)malloc(sizeof(big));
	m = (big*)malloc(banyakdata * sizeof(big));
	k = (big*)malloc(banyakdata * sizeof(big));
	res = (big*)malloc(banyakdata * 2 * sizeof(big));

	init(p,g,x,e,y,m,k,res);
	mainenkripsi(m,k,res,g,p,y);

	free(p->value);
	free(p);
	free(g->value);
	free(g);
	free(x->value);
	free(x);
	free(e->value);
	free(e);
	free(y->value);
	free(y);
	free(m->value);
	free(m);
	free(k->value);
	free(k);
	free(res->value);
	free(res);
	
	return 0;
}

