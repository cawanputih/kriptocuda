#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ul;
typedef unsigned int uint;

int banyakdata = 1024;
int dimensigrid = 8;
int dimensiblok = 128;

typedef struct {
	char size;
	uint* value;
}big;

typedef struct {
	short size;
	char* value;
}stringnumber;


__host__ __device__ short ukuranbit(big *a);
__host__ __device__ char getbit(big* a, short count);
__host__ __device__ uint getShiftedBlock(big *num, char noblok, char geser);
__host__ __device__ void kali(big *a, big *b, big* res);
__host__ __device__ void modulo(big* a, big* b, big* res, uint* minbuff);
__host__ __device__ void modexp(big* a, big* b, big* c, big* res, uint* minbuff, big* mulbuff);
__device__ void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff);
__device__ void dekripsi(big *c1, big *c2, big *e, big *p, big *res, big *minbuff, big *mulbuff);
__global__ void kernelenk(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff);
__global__ void kerneldek(big *c, big *e, big *p, big *res, big *minbuff, big *mulbuff);
cudaError_t CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res);
cudaError_t CUDAdek(big *c, big *e, big* p, big *res);
void mainenkripsi(big *m, big *k, big *res, big *g, big *p, big *y);
void maindekripsi(big* c,big* x,big* p,big* res2);
void tambah(big* a, char b, big* res);
void kurang(big* a, big *b, big* res);
void divandmod(big* a, big* &b, big* divres, big* modres, uint* minbuff);
void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff);
void init(big *p, big *g, big *x, big*e, big *y, big *m, big *k, big *res, big *res2);
void copybig(big* a, big* res);
void stringtobig(stringnumber* sn, big* res, big* mulbuff, big* ten);
void bigtostring(big* x, stringnumber* sn, big* ten, big* xbuff, big* divbuff, big* modbuff, uint* minbuff);
void printsn(stringnumber* sn);
void teskonversi();


__device__ void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff) {
	// BLok 1 Cipher
	modexp(g,k,p,res,minbuff->value,mulbuff);
	// Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff->value,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff->value);
}

__device__ void dekripsi(big *c1, big *c2, big *e, big *p, big *res, big *minbuff, big *mulbuff) {
	modexp(c1,e,p,res,minbuff->value,mulbuff);
	kali(res, c2, mulbuff);
	modulo(mulbuff, p, res, minbuff->value);
}

__global__ void kernelenk(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	enkripsi(m + i, k + i, g, p, y, res + 2 * i, minbuff+i, mulbuff+i);
}

__global__ void kerneldek(big *c, big *e, big *p, big *res, big *minbuff, big *mulbuff){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	dekripsi(c + 2*i, c + 2*i+1, e, p, res+i, minbuff+i, mulbuff+i);
}

cudaError_t CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res) {
	cudaError_t cudaStatus;
	cudaSetDevice(0);
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
		cudaMalloc((void**)&tempvalue[i], (sizeof(uint) * m[0].size));
		cudaMemcpy(tempvalue[i], m[0].value, (sizeof(uint) * m[0].size), cudaMemcpyHostToDevice);
		temp.size = m[0].size;
		temp.value = tempvalue[i];
		cudaMemcpy((devm + i), &temp, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp2;
		cudaMalloc((void**)&tempvalue2[i], (sizeof(uint) * k[0].size));
		cudaMemcpy(tempvalue2[i], k[0].value, (sizeof(uint) * k[0].size), cudaMemcpyHostToDevice);
		temp2.size = k[0].size;
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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernelenk << <dimensigrid, dimensiblok >> >(devm, devk, devg, devp, devy, devres, minbuff, mulbuff);

	cudaStatus = cudaGetLastError();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Durasi = %f milidetik\n", milliseconds);
	// printf("GPU Memory: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	else {
		// printf("Success\n");
	}

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

	return cudaStatus;
}

cudaError_t CUDAdek(big *c, big *e, big* p, big *res) {
	cudaError_t cudaStatus;
	cudaSetDevice(0);
	//=====================BAGIAN p dan e ( eksponen)  ====================================//
	big *devp, *deve;

	cudaMalloc((void**)&devp, sizeof(big));
	cudaMalloc((void**)&deve, sizeof(big));

	uint *darrp, *darre;
	cudaMalloc((void**)&darrp, p->size * sizeof(uint));
	cudaMalloc((void**)&darre, e->size * sizeof(uint));

	big tempp;
	cudaMemcpy(darrp, p->value, (sizeof(uint) * p->size), cudaMemcpyHostToDevice);
	tempp.size = p->size;
	tempp.value = darrp;
	cudaMemcpy((devp), &tempp, (sizeof(big)), cudaMemcpyHostToDevice);

	big tempe;
	cudaMemcpy(darre, e->value, (sizeof(uint) * e->size), cudaMemcpyHostToDevice);
	tempe.size = e->size;
	tempe.value = darre;
	cudaMemcpy((deve), &tempe, (sizeof(big)), cudaMemcpyHostToDevice);

	//======================================BAGIAN C[] ====================================//
	big *devc, *devres, *minbuff, *mulbuff;

	cudaMalloc((void**)&devc, banyakdata * 2 * sizeof(big));
	cudaMalloc((void**)&devres, banyakdata  * sizeof(big));
	cudaMalloc((void**)&minbuff, banyakdata  * sizeof(big));
	cudaMalloc((void**)&mulbuff, banyakdata  * sizeof(big));

	uint **tempvalue = (uint**)malloc(sizeof(uint*)*banyakdata*2);
	uint **tempvalue2 = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue3 = (uint**)malloc(sizeof(uint*)*banyakdata);
	uint **tempvalue4 = (uint**)malloc(sizeof(uint*)*banyakdata);

	// Alokasi Memori untuk blok m dan k
	for (int i = 0; i < banyakdata; i++) {
		big temp11;
		cudaMalloc((void**)&tempvalue[2*i], (sizeof(uint) * c[0].size));
		cudaMemcpy(tempvalue[2*i], c[0].value, (sizeof(uint) * c[0].size), cudaMemcpyHostToDevice);
		temp11.size = c[0].size;
		temp11.value = tempvalue[2*i];
		cudaMemcpy((devc + 2*i), &temp11, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp12;
		cudaMalloc((void**)&tempvalue[2*i+1], (sizeof(uint) * c[1].size));
		cudaMemcpy(tempvalue[2*i+1], c[1].value, (sizeof(uint) * c[1].size), cudaMemcpyHostToDevice);
		temp12.size = c[1].size;
		temp12.value = tempvalue[2*i+1];
		cudaMemcpy((devc + 2*i+1), &temp12, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp2;
		cudaMalloc((void**)&tempvalue2[i], (sizeof(uint) * p->size * 2));
		temp2.value = tempvalue2[i];
		cudaMemcpy((devres + i), &temp2, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp3;
		cudaMalloc((void**)&tempvalue3[i], (sizeof(uint) * p->size * 2));
		temp3.value = tempvalue3[i];
		cudaMemcpy((minbuff + i), &temp3, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp4;
		cudaMalloc((void**)&tempvalue4[i], (sizeof(uint) * p->size * 2));
		temp4.value = tempvalue4[i];
		cudaMemcpy((mulbuff + i), &temp4, (sizeof(big)), cudaMemcpyHostToDevice);
	}


	// size_t free_byte ;
 //    size_t total_byte ;
 //    cudaMemGetInfo( &free_byte, &total_byte ) ;
	// double free_db = (double)free_byte ;
 //    double total_db = (double)total_byte ;
 //    double used_db = total_db - free_db ;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kerneldek << <dimensigrid, dimensiblok >> >(devc, deve, devp, devres, minbuff, mulbuff);

	cudaStatus = cudaGetLastError();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Durasi = %f milidetik\n", milliseconds);
	// printf("GPU Memory: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	else {
		// printf("Success\n");
	}

	cudaDeviceSynchronize();


	//	COPY FROM DEVICE TO HOST HERE 

	big* tempres = (big*) malloc(banyakdata * sizeof(big)); 
	for (int i = 0; i < banyakdata; i++){
		tempres[i].value = (uint*) malloc(sizeof(uint) * p->size);
	}
	cudaMemcpy(tempres, devres, (sizeof(big) * banyakdata), cudaMemcpyDeviceToHost);

	for (int i = 0; i < banyakdata; i++){
		res[i].size = tempres[i].size;
		cudaMemcpy(res[i].value, tempres[i].value, sizeof(uint) * p->size, cudaMemcpyDeviceToHost);
	}

	cudaFree(darrp);
	cudaFree(darre);
	cudaFree(devp);
	cudaFree(deve);

	for (int i = 0; i < 2 * banyakdata; i++) {
		cudaFree(tempvalue[i]);
	}

	for (int i = 0; i < banyakdata; i++) {
		cudaFree(tempvalue2[i]);
		cudaFree(tempvalue3[i]);
		cudaFree(tempvalue4[i]);
	}

	free(tempvalue);
	free(tempvalue2);
	free(tempvalue3);
	free(tempvalue4);
	cudaFree(devc);
	cudaFree(devres);
	cudaFree(minbuff);
	cudaFree(mulbuff);

	free(tempres);
    
	return cudaStatus;
}

void mainenkripsi(big *m, big *k, big *res, big *g, big *p, big *y){
	printf("Encrypting...\n");
	//========================================================//
	cudaError_t cudaStatus = CUDAenk(m, k, g, p, y, res);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nenkripsiCUDA failed!");
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	for (int i = 0; i < 5; i++)
	{
		printf("Cipher %d  size %d : %u\n",i, res[i].size, res[i].value[0]);
	}
	printf("Cipher ... : ...\n");
	printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);
}

void maindekripsi(big* c,big* e,big* p,big* res2){
	printf("Decrypting...\n");
	//========================================================//

	cudaError_t cudaStatus = CUDAdek(c, e, p, res2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ndekripsiCUDA failed!");
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	for (int i = 0; i < 5; i++)
	{
		printf("Plain %d  size %d : %u\n",i, res2[i].size, res2[i].value[0]);
		printf("Plain %d  size %d : %u\n",i, res2[i].size, res2[i].value[1]);
	}
	printf("Plain ... : ...\n");
	printf("Plain %d  size %d : %u\n",banyakdata-1, res2[banyakdata-1].size, res2[banyakdata-1].value[0]);
}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

void init(big *p, big *g, big *x, big*e, big *y, big *m, big *k, big *res, big *res2){
	// Kunci publik p
	p->size = 12;
	p->value = (uint*) malloc(p->size * sizeof(uint));
	for (int i = 0; i < p->size; i++)
	{
		// p->value[i] = 2357;
		p->value[i] = UINT_MAX-82;
	}
	// p->value[0] = UINT_MAX-4;
	// p->value[0] = 2387;
	// p->value[1] = 2357;


	// Kunci publik g
	g->size = 12;
	g->value = (uint*) malloc(g->size * sizeof(uint));
	for (int i = 0; i < g->size; i++)
	{
		// g->value[i] = 2;
		g->value[i] = UINT_MAX-902;
	}

	// Kunci privat x
	x->size = 12;
	x->value = (uint*) malloc(x->size * sizeof(uint));
	for (int i = 0; i < x->size; i++)
	{
		// x->value[i] = 1751;
		x->value[i] = UINT_MAX-86262;
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
	printf("y 0 : %u\n", y->value[0]);
	printf("y 0 : %u\n", y->value[1]);
	//========================================================//
	// Blok plainteks
	m->size = 12;
	m->value = (uint*) malloc(m->size * sizeof(uint));
	for (int i = 0; i < m->size; i++)
	{
		// m->value[i] = 1001;
		m->value[i] = UINT_MAX-5522;
	}

	// Nilai k masing-masing blok
	k->size = 12;
	k->value = (uint*) malloc(k->size * sizeof(uint));
	for (int i = 0; i < k->size; i++)
	{
		// k->value[i] = 77;
		k->value[i] = UINT_MAX-38227;
	}

	// Alokasi memori untuk result
	for (int i = 0; i < banyakdata*2; i++)
	{
		res[i].value = (uint*) malloc(sizeof(uint) * p->size);
	}

	// Alokasi memori untuk result 2
	for (int i = 0; i < banyakdata; i++)
	{
		res2[i].value = (uint*) malloc(sizeof(uint) * p->size);
	}
}

int main(){
	big *p, *g, *x, *e, *y, *m, *k, *res, *res2;
	p = (big*)malloc(sizeof(big));
	g = (big*)malloc(sizeof(big));
	x = (big*)malloc(sizeof(big));
	e = (big*)malloc(sizeof(big));
	y = (big*)malloc(sizeof(big));
	m = (big*)malloc(banyakdata * sizeof(big));
	k = (big*)malloc(banyakdata * sizeof(big));
	res = (big*)malloc(banyakdata * 2 * sizeof(big));
	res2 = (big*)malloc(banyakdata * sizeof(big));

	init(p,g,x,e,y,m,k,res,res2);
	mainenkripsi(m,k,res,g,p,y);
	printf("			=========================			\n");
	maindekripsi(res,e,p,res2);


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
	free(res2->value);
	free(res2);

	//teskonversi();
	return 0;
}


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

void divandmod(big* a, big* &b, big* divres, big* modres, uint* minbuff) {
	modres->size = a->size;
	for(char i = 0 ; i < modres->size ;i++){
		modres->value[i] = a->value[i];
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

	modres->value[modres->size] = 0;
	modres->size++;

	divres->size = ukurana - ukuranb + 1;
	for (i = 0; i < divres->size; i++)
		divres->value[i] = 0;

	i = ukurana - ukuranb + 1;
	while (i > 0) {
		i--;
		divres->value[i] = 0;
		i2 = 32;
		while (i2 > 0) {
			i2--;
			for (j = 0, k = i, borrowIn = 0; j <= ukuranb; j++, k++) {
				temp = modres->value[k] - getShiftedBlock(b, j, i2);
				borrowOut = (temp > modres->value[k]);
				if (borrowIn) {
					borrowOut |= (temp == 0);
					temp--;
				}
				minbuff[k] = temp; 
				borrowIn = borrowOut;
			}

			for (; k < ukurana && borrowIn; k++) {
				borrowIn = (modres->value[k] == 0);
				minbuff[k] = modres->value[k] - 1;
			}

			if (!borrowIn) {
				divres->value[i] |= ((uint) 1  << i2);
				while (k > i) {
					k--;
					modres->value[k] = minbuff[k];
				}
			} 
		}
	}

	if (divres->value[divres->size - 1] == 0)
		divres->size--;

	while (modres->size > 0 && modres->value[modres->size - 1] == 0)
		modres->size--;
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

void copybig(big* a, big* res){
	res->size = a->size;
	for (int i = 0; i < res->size; i++){
		res->value[i] = a->value[i];
	}
}

void stringtobig(stringnumber* sn, big* res, big* mulbuff, big* ten){
	res->size = 0;
	for (int i = sn->size-1; i >= 0; i--){
		kali(res, ten, mulbuff);
		tambah(mulbuff, sn->value[i], res);
	}
}

void bigtostring(big* x, stringnumber* sn, big* ten, big* xbuff, big* divbuff, big* modbuff, uint* minbuff) {

	copybig(x,xbuff);
	short snlength = 0;

	while (xbuff->size != 0 ) {
		divandmod(xbuff,ten,divbuff,modbuff,minbuff);
		sn->value[snlength] = (char) modbuff->value[0];
		snlength++;
		copybig(divbuff,xbuff);
	}

	sn->size = snlength;
}

void printsn(stringnumber* sn){
	for (int i = 0; i < sn->size; ++i){
		printf("%d", sn->value[sn->size-i-1]);
	}
	printf("\n");
}

void teskonversi(){
	int seed = time(NULL);
    srand(seed);

	stringnumber *sn = (stringnumber*) malloc(sizeof(stringnumber));
	sn->size = 25;
	sn->value = (char *) malloc(sn->size);

	for (int i = 0; i < sn->size; i++)
	{
		sn->value[i] = rand() % 10;
	}

	big* konversi = (big*) malloc(sizeof(big));
	big* mulbuff = (big*) malloc(sizeof(big));
	big* ten = (big*) malloc(sizeof(big));

	konversi->value = (uint*) malloc(sizeof(10));
	mulbuff->value = (uint*) malloc(sizeof(10));
	ten->value = (uint*) malloc(sizeof(1));
	ten->size = 1;
	ten->value[0] = 10;

	printf("Stringnumber awal : ");
	printsn(sn);
	stringtobig(sn, konversi, mulbuff, ten);
	printf("konversi size %d\n", konversi->size);
	printf("konversi value 0  %u\n", konversi->value[0]);
	printf("konversi value 0  %u\n", konversi->value[1]);

	stringnumber *sn2 = (stringnumber*) malloc(sizeof(stringnumber));
	big* xbuff = (big*) malloc(sizeof(big));
	big* divbuff = (big*) malloc(sizeof(big));
	big* modbuff = (big*) malloc(sizeof(big));

	sn2->value = (char *) malloc(100);

	xbuff->value = (uint *) malloc(sizeof(uint) * 10);
	divbuff->value = (uint *) malloc(sizeof(uint) * 10);
	modbuff->value = (uint *) malloc(sizeof(uint) * 10);

	uint* minbuff = (uint*) malloc(sizeof(uint) * 10);

	bigtostring(konversi,sn2,ten,xbuff,divbuff,modbuff,minbuff);
	printf("Stringnumber akhir : ");
	printsn(sn2);
}
