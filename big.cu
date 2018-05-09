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
void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff);
void dekripsi(big *c1, big *c2, big *e, big *p, big *res, big *minbuff, big *mulbuff);
void kernelenk(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff);
void kerneldek(big *c, big *e, big *p, big *res, big *minbuff, big *mulbuff);
void CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res);
void CUDAdek(big *c, big *e, big* p, big *res);
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


void enkripsi(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff) {
	// BLok 1 Cipher
	modexp(g,k,p,res,minbuff->value,mulbuff);
	// Blok 2 Cipher
	modexp(y, k, p, res + 1,minbuff->value,mulbuff);
	kali(res + 1, m, mulbuff);
	modulo(mulbuff, p, res+1, minbuff->value);
}

void dekripsi(big *c1, big *c2, big *e, big *p, big *res, big *minbuff, big *mulbuff) {
	modexp(c1,e,p,res,minbuff->value,mulbuff);
	kali(res, c2, mulbuff);
	modulo(mulbuff, p, res, minbuff->value);
}

void kernelenk(big *m, big *k, big *g, big *p, big *y, big *res, big *minbuff, big *mulbuff){
	for (int i = 0; i < banyakdata; i++)
	{
		enkripsi(m + i, k + i, g, p, y, res + 2 * i, minbuff+i, mulbuff+i);
	}
}

void kerneldek(big *c, big *e, big *p, big *res, big *minbuff, big *mulbuff){
	for (int i = 0; i < banyakdata; i++)
	{
		dekripsi(c + 2*i, c + 2*i+1, e, p, res+i, minbuff+i, mulbuff+i);
	}
}

void CUDAenk(big *m, big *k, big* g, big* p, big* y, big *res) {
	big *minbuff, *mulbuff;
	minbuff = (big*) malloc(banyakdata  * sizeof(big));
	mulbuff = (big*) malloc(banyakdata  * sizeof(big));

	for (int i = 0; i < banyakdata; i++) {
		minbuff[i].value = (uint*) malloc(sizeof(uint) * p->size * 2);
		mulbuff[i].value = (uint*) malloc(sizeof(uint) * p->size * 2);
	}


	clock_t begin = clock();
		kernelenk(m, k, g, p, y, res, minbuff, mulbuff);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / 1000;
	printf("Durasi  : %f ms\n", time_spent);
}

void CUDAdek(big *c, big *e, big* p, big *res) {
	big *minbuff, *mulbuff;
	minbuff = (big*) malloc(banyakdata  * sizeof(big));
	mulbuff = (big*) malloc(banyakdata  * sizeof(big));

	for (int i = 0; i < banyakdata; i++) {
		minbuff[i].value = (uint*) malloc(sizeof(uint) * p->size * 2);
		mulbuff[i].value = (uint*) malloc(sizeof(uint) * p->size * 2);
	}

	clock_t begin = clock();
		kerneldek(c, e, p, res, minbuff, mulbuff);
	clock_t end = clock();
	double time_spent = (double)(end - begin) / 1000;
	printf("Durasi  : %f ms\n", time_spent);
}

void mainenkripsi(big *m, big *k, big *res, big *g, big *p, big *y){
	printf("Encrypting...\n");
	CUDAenk(m, k, g, p, y, res);
	
	for (int i = 0; i < 5; i++)
	{
		printf("Cipher %d  size %d : %u\n",i, res[i].size, res[i].value[0]);
	}
	printf("Cipher ... : ...\n");
	printf("Cipher %d  size %d : %u\n",banyakdata*2-2, res[banyakdata*2-2].size, res[banyakdata*2-2].value[0]);
	printf("Cipher %d  size %d : %u\n",banyakdata*2-1, res[banyakdata*2-2].size, res[banyakdata*2-1].value[0]);
}

void maindekripsi(big* c, big* e,big* p,big* res2){
	printf("Decrypting...\n");
	
	CUDAdek(c, e, p, res2);

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
		//p->value[i] = 2357;
		p->value[i] = UINT_MAX-82;;
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
	// printf("e adalah %u\n", e->value[0]);
	free(xplus1->value);
	free(xplus1);

	// Cari nilai kunci publik y = (g^x) mod p
	big* mulbuff = (big*) malloc(sizeof(big));
	mulbuff->value = (uint*) malloc(sizeof(uint) * p->size * 2);
	uint* minbuff = (uint*) malloc(sizeof(uint) * p->size * 2);

	y->value = (uint*) malloc(p->size * 2 * sizeof(uint));
	carikunciy(g,x,p,y,minbuff,mulbuff);
	// printf("y adalah %u\n",y->value[0]);
	//========================================================//
	// Blok plainteks
	for(int i = 0 ; i < banyakdata ; i++){
		m[i].size = 12;
		m[i].value = (uint*) malloc(m[i].size * sizeof(uint));
		for (int j = 0; j < m[i].size; j++)
		{
			// m[i].value[j] = 1001;
			m[i].value[j] = UINT_MAX-5522;
		}

		// Nilai k masing-masing blok
		k[i].size = 12;
		k[i].value = (uint*) malloc(k[i].size * sizeof(uint));
		for (int j = 0; j < k[i].size; j++)
		{
			// k[i].value[j] = 77;
			k[i].value[j] = UINT_MAX-38227;
		}
	}
	// Alokasi memori untuk result
	for (int i = 0; i < banyakdata*2; i++)
	{
		res[i].value = (uint*) malloc(sizeof(uint) * p->size *2);
	}

	// Alokasi memori untuk result 2
	for (int i = 0; i < banyakdata; i++)
	{
		res2[i].value = (uint*) malloc(sizeof(uint) * p->size * 2);
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
