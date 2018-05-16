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

__global__ void kernelenk(big *m, big *k, big *g, big *p, big *res){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int jdx = threadIdx.x;

	__shared__ big sm[128];
	__shared__ big sk[128];
	__shared__ big smulbuff[128];
	__shared__ big sres[256];
	__shared__ big sp;
	__shared__ big sg;
	__shared__ big sy;
	__shared__ uint s[2400];

	uint *sresval = s;
	uint *smulbuffval = (uint*)&sresval[4*128*2];
	uint *sminbuffval = (uint*)&smulbuffval[4*128];
	uint *spval = (uint*)&sminbuffval[2*128];
	uint *sgval = (uint*)&spval[2];
	uint *syval = (uint*)&sgval[2];
	uint *smval = (uint*)&syval[2];
	uint *skval = (uint*)&smval[2*128];


	sm[jdx].size = m[idx].size;
	sk[jdx].size = k[idx].size;
	sp.size = p[0].size;
	sg.size = g[0].size;

	for (int i = 0; i < 2; i++)
	{
		smval[jdx*2+i] = m[idx].value[i];
		skval[jdx*2+i] = k[idx].value[i];
		spval[i] = p[0].value[i];
		sgval[i] = g[0].value[i];
	}

	sm[jdx].value = (uint*)&smval[jdx*2];
	sk[jdx].value = (uint*)&skval[jdx*2];
	sres[2*jdx].value = (uint*)&sresval[jdx*4*2];
	sres[2*jdx+1].value = (uint*)&sresval[jdx*4*2+4];
	smulbuff[jdx].value = (uint*)&smulbuffval[jdx*4];
	sp.value = spval;
	sg.value = sgval;

	__syncthreads();

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


void CUDAenk(big *m, big *k, big* g, big* p, big *res) {
	
	//=====================BAGIAN G, P, DAN Y ====================================//
	big *devg, *devp;

	cudaMalloc((void**)&devg, sizeof(big));
	cudaMalloc((void**)&devp, sizeof(big));

	uint *darrg, *darrp;
	cudaMalloc((void**)&darrg, g->size * sizeof(uint));
	cudaMalloc((void**)&darrp, p->size * sizeof(uint));
	

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

	kernelenk << <dimensigrid, dimensiblok >> >(devm, devk, devg, devp, devres);

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
	cudaFree(devg);
	cudaFree(devp);

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

void mainenkripsi(big *m, big *k, big *res, big *g, big *p){
	// printf("Encrypting...\n");
	//========================================================//

	cudaSetDevice(0);

	CUDAenk(m, k, g, p, res);

	cudaDeviceReset();

	// for (int i = 0; i < 5; i++)
	// {
	// 	printf("Cipher %d  size %d : %u\n",i, res[i].size, res[i].value[0]);
	// }
	// printf("Cipher ... : ...\n");
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	// printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);
}

void init(big *p, big *g, big *x, big *m, big *k, big *res){
	// Kunci publik p
	srand(2018);

	p->size = 12;
	p->value = (uint*) malloc(p->size * sizeof(uint));
	p->value[0] = UINT_MAX;
	for (int i = 1; i < p->size; i++)
	{
		p->value[i] = rand() % UINT_MAX;
	}


	// Kunci publik g
	g->size = 12;
	g->value = (uint*) malloc(g->size * sizeof(uint));
	for (int i = 0; i < g->size; i++)
	{
		// g->value[i] = 2;
		g->value[i] = rand() % UINT_MAX;
	}

	// Kunci privat x
	x->size = 12;
	x->value = (uint*) malloc(x->size * sizeof(uint));
	for (int i = 0; i < x->size; i++)
	{
		// x->value[i] = 1751;
		x->value[i] = rand() % UINT_MAX;
	}

	//========================================================//
	// Blok plainteks
	for(int i = 0 ; i < banyakdata ; i++){
		m[i].size = 12;
		m[i].value = (uint*) malloc(m[i].size * sizeof(uint));
		for (int j = 0; j < m[i].size; j++)
		{
			m[i].value[j] = rand() % UINT_MAX;
		}

		// Nilai k masing-masing blok
		k[i].size = 12;
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
	big *p, *g, *x, *m, *k, *res;
	p = (big*)malloc(sizeof(big));
	g = (big*)malloc(sizeof(big));
	x = (big*)malloc(sizeof(big));
	m = (big*)malloc(banyakdata * sizeof(big));
	k = (big*)malloc(banyakdata * sizeof(big));
	res = (big*)malloc(banyakdata * 2 * sizeof(big));

	init(p,g,x,m,k,res);
	mainenkripsi(m,k,res,g,p);

	free(p->value);
	free(p);
	free(g->value);
	free(g);
	free(x->value);
	free(x);
	free(m->value);
	free(m);
	free(k->value);
	free(k);
	free(res->value);
	free(res);
	
	return 0;
}
