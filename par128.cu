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
void CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res);
void CUDAdek(big *c, big *e, big* p, big *res);
void mainenkripsi(big *m, big *k, big *res, big *g, big *p, big *y);
void maindekripsi(big* c,big* x,big* p,big* res2);
void tambah(big* a, char b, big* res);
void kurang(big* a, big *b, big* res);
void divandmod(big* a, big* &b, big* divres, big* modres, uint* minbuff);
void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff);
void init(big *p, big *g, big *x, big*e, big *y, big *m, big *k, big *res, big *res2);


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
	// // BLok 1 Cipher
	modexp(g,k,p,res,minbuff->value,mulbuff);
	// printf("res adalah %u\n", res->value[0]);
	// // Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff->value,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff->value);

	// printf("res  val 0 adalah %p\n", &(res->value[0]));
	// printf("res  val 1 adalah %p\n", &(res->value[1]));
	// printf("res  val 2 adalah %p\n", &(res->value[2]));
	// printf("res  val 3 adalah %p\n", &(res->value[3]));

	// printf("res 1 val 0 adalah %p\n", &((res+1)->value[0]));
	// printf("res 1 val 1 adalah %p\n", &((res+1)->value[1]));
	// printf("res 1 val 2 adalah %p\n", &((res+1)->value[2]));
	// printf("res 1 val 3 adalah %p\n", &((res+1)->value[3]));

	// printf("res val 0 adalah %u\n", res->value[0]);
	// printf("res 1 val 0 adalah %u\n", (res+1)->value[0]);

}

__device__ void dekripsi(big *c1, big *c2, big *e, big *p, big *res, big *minbuff, big *mulbuff) {
	modexp(c1,e,p,res,minbuff->value,mulbuff);
	kali(res, c2, mulbuff);
	modulo(mulbuff, p, res, minbuff->value);
	// printf("c1 adlaah %u\n", c1->value[0]);
	// printf("c2 adlaah %u\n", c2->value[0]);
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
	__shared__ uint s[3200];

	uint *sresval = s;
	uint *spval = (uint*)&sresval[8*128*2];
	uint *sgval = (uint*)&spval[4];
	uint *syval = (uint*)&sgval[4];
	uint *smval = (uint*)&syval[4];
	uint *skval = (uint*)&smval[4*128];


	sm[jdx].size = m[idx].size;
	sk[jdx].size = k[idx].size;
	sp.size = p[0].size;
	sg.size = g[0].size;
	sy.size = y[0].size;

	for (int i = 0; i < 4; i++)
	{
		smval[jdx*4+i] = m[idx].value[i];
		skval[jdx*4+i] = k[idx].value[i];
		spval[i] = p[0].value[i];
		sgval[i] = g[0].value[i];
		syval[i] = y[0].value[i];
	}

	sm[jdx].value = (uint*)&smval[jdx*4];
	sk[jdx].value = (uint*)&skval[jdx*4];
	sres[2*jdx].value = (uint*)&sresval[jdx*8*2];
	sres[2*jdx+1].value = (uint*)&sresval[jdx*8*2+8];
	sp.value = spval;
	sg.value = sgval;
	sy.value = syval;

	__syncthreads();

	// if(idx < 10){
	// 		// printf("pointer2 di %d = %p \n", 2*jdx,sres[2*jdx].value);
	// 		// printf("pointer2 di %d = %p \n", 2*jdx+1,sres[2*jdx+1].value);

	// 		// printf("sresval pointer di %d = %p \n", jdx,sresval + jdx);

	// 		// printf("pointer big di %d = %p \n", 2*jdx,sres+2*jdx);
	// 		// printf("pointer big2 di %d = %p \n", 2*jdx+1,sres+2*jdx+1);

	big* minbuff = (big*) malloc(sizeof(big));
	big* mulbuff = (big*) malloc(sizeof(big));

	minbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);
	mulbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);


			enkripsi(sm + jdx, sk + jdx, &sg, &sp, &sy, sres + 2*jdx, minbuff, mulbuff);
	// }

	// printf("sres %d adalah %d\n", 2*idx, sres[2*jdx].size);
	// printf("sres %d adalah %d\n", 2*idx+1, sres[2*jdx+1].size);

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

__global__ void kerneldek(big *c, big *e, big *p, big *res){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	__shared__ big sa[128];
	__shared__ big sb[128];
	__shared__ big sres[128];
	__shared__ big sp;
	__shared__ big se;
	__shared__ uint s[2100];

	uint *sresval = s;
	uint *spval = (uint*)&sresval[8*128];
	uint *saval = (uint*)&spval[4];
	uint *sbval = (uint*)&saval[4*128];
	uint *seval = (uint*)&sbval[4*128];

	sa[jdx].size = c[2*idx].size;
	sb[jdx].size = c[2*idx+1].size;
	sp.size = p[0].size;
	se.size = e[0].size;

	for (int i = 0; i < 4; i++)
	{
		saval[jdx*4+i] = c[2*idx].value[i];
		sbval[jdx*4+i] = c[2*idx+1].value[i];
		spval[i] = p[0].value[i];
		seval[i] = e[0].value[i];
	}

	sa[jdx].value = (uint*)&saval[jdx*4];
	sb[jdx].value = (uint*)&sbval[jdx*4];
	sres[jdx].value = (uint*)&sresval[jdx*8];
	sp.value = spval;
	se.value = seval;

	__syncthreads();

	big* minbuff = (big*) malloc(sizeof(big));
	big* mulbuff = (big*) malloc(sizeof(big));

	minbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);
	mulbuff->value = (uint*) malloc(sizeof(uint) * sp.size * 2);

		dekripsi(sa+jdx, sb+jdx, &se, &sp, sres+jdx, minbuff, mulbuff);

	res[idx].size = sres[jdx].size;

	for (int i = 0; i < sres[jdx].size; i++)
	{
		res[idx].value[i] = sres[jdx].value[i];
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

void CUDAdek(big *c, big *e, big* p, big *res) {
	
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
		cudaMalloc((void**)&tempvalue[2*i], (sizeof(uint) * c[2*i].size));
		cudaMemcpy(tempvalue[2*i], c[2*i].value, (sizeof(uint) * c[2*i].size), cudaMemcpyHostToDevice);
		temp11.size = c[2*i].size;
		temp11.value = tempvalue[2*i];
		cudaMemcpy((devc + 2*i), &temp11, (sizeof(big)), cudaMemcpyHostToDevice);

		big temp12;
		cudaMalloc((void**)&tempvalue[2*i+1], (sizeof(uint) * c[2*i+1].size));
		cudaMemcpy(tempvalue[2*i+1], c[2*i+1].value, (sizeof(uint) * c[2*i+1].size), cudaMemcpyHostToDevice);
		temp12.size = c[2*i+1].size;
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

	
	kerneldek << <dimensigrid, dimensiblok >> >(devc, deve, devp, devres);

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

void maindekripsi(big* c,big* e,big* p,big* res2){
	// printf("Decrypting...\n");
	//========================================================//

	cudaSetDevice(0);

	CUDAdek(c, e, p, res2);

	cudaDeviceReset();

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Plain %d  size %d : %u\n",i, res2[i].size, res2[i].value[0]);
	// 	printf("Plain %d  size %d : %u\n",i, res2[i].size, res2[i].value[1]);
	// }
	// printf("Plain ... : ...\n");
	// printf("Plain %d  size %d : %u\n",banyakdata-1, res2[banyakdata-1].size, res2[banyakdata-1].value[0]);
}

void carikunciy(big *g, big *x, big *p, big *y, uint *minbuff, big *mulbuff){

	modexp(g,x,p,y,minbuff,mulbuff);
}

void init(big *p, big *g, big *x, big*e, big *y, big *m, big *k, big *res, big *res2){
	// Kunci publik p
	srand(2018);

	p->size = 4;
	p->value = (uint*) malloc(p->size * sizeof(uint));
	p->value[0] = UINT_MAX;
	for (int i = 1; i < p->size; i++)
	{
		p->value[i] = rand() % UINT_MAX;
	}


	// Kunci publik g
	g->size = 4;
	g->value = (uint*) malloc(g->size * sizeof(uint));
	for (int i = 0; i < g->size; i++)
	{
		// g->value[i] = 2;
		g->value[i] = rand() % UINT_MAX;
	}

	// Kunci privat x
	x->size = 4;
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
		m[i].size = 4;
		m[i].value = (uint*) malloc(m[i].size * sizeof(uint));
		for (int j = 0; j < m[i].size; j++)
		{
			m[i].value[j] = rand() % UINT_MAX;
		}

		// Nilai k masing-masing blok
		k[i].size = 4;
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
	// printf("			=========================			\n");
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