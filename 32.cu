#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 102400;
int dimensigrid = 800;
int dimensiblok = 128;

void modexp(ulint a, ulint b, ulint c, ulint* res) {
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

void enkripsi(ulint g, ulint k, ulint p, ulint m, ulint y, ulint *res) {
	modexp(g, k, p, res);
	modexp(y, k, p, res + 1);
	
	*(res + 1) = *(res + 1) * m % p;
}

void dekripsi(ulint a, ulint b, ulint p, ulint e, ulint *res) {
	modexp(a, e, p, res);
	*res = *res * b % p;
}

void kernelenk(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		enkripsi(g, k[i], p, m[i], y, res + 2 * i);
	}
}

void kerneldek(ulint *c, ulint p, ulint e, ulint *res) {
	for (int i = 0; i < banyakdata; i++)
	{
		dekripsi(c[2*i], c[2*i+1], p, e, res + i);
	}
}

void enkripsiCUDA(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	clock_t begin = clock();
		kernelenk(m,k,g,p,y,res);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
	printf("\n<<<<<<<<<<<<<<HASIL KE CPU>>>>>>>>>>>>>>>\n");
}

void dekripsiCUDA(ulint *c, ulint p, ulint e, ulint *res2) {
	clock_t begin = clock();
		kerneldek(c,p,e,res2);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent/1000);
	printf("\n<<<<<<<<<<<<<<HASIL KE CPU>>>>>>>>>>>>>>>\n");
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

    //printf("count : %d\n", indexpesan);

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
	//initenkripsi2(m, k);


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
	//writedekrip(res2);

	free(m);
	free(k);
	free(res);
	free(res2);

	return 0;
}