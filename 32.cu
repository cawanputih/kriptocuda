#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef unsigned long ulint;
typedef unsigned long long ulint64;

int banyakdata = 256;
int dimensigrid = 2;
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
		dekripsi(c[2 * i], c[2 * i + 1], p, e, res + i);
	}
}

void enkripsiCUDA(ulint *m, ulint *k, ulint g, ulint p, ulint y, ulint *res) {
	clock_t begin = clock();
	kernelenk(m, k, g, p, y, res);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent / 1000);
}

void dekripsiCUDA(ulint *c, ulint p, ulint e, ulint *res2) {
	clock_t begin = clock();
	kerneldek(c, p, e, res2);
	clock_t end = clock();

	double time_spent = (double)(end - begin);
	printf("Durasi  : %f milliseconds\n", time_spent / 1000);
}

void initenkripsi(ulint *m, ulint *k) {

	for (int i = 0; i < banyakdata; i++) {
		m[i] = rand() % 3999999978;
		k[i] = rand() % 3999999978;
	}
}

int main() {
	ulint *m, *k, *res, *res2, g, p, y, x, e;

	m = (ulint*)malloc(banyakdata * sizeof(ulint));
	k = (ulint*)malloc(banyakdata * sizeof(ulint));
	res = (ulint*)malloc(banyakdata * 2 * sizeof(ulint));
	res2 = (ulint*)malloc(banyakdata * sizeof(ulint));

	srand(2018);

	g = rand() % 3999999978;
	p = 3999999979;
	x = rand() % 3999999978;
	modexp(g, x, p, &y);
	initenkripsi(m, k);

	// printf("<<<<<<<<<<<<<<Pesan Asli>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("m[%d] = %lu\n", i, m[i]);
	// }

	// printf("m[...]\n");
	// printf("m[%d] = %lu\n", banyakdata - 1, m[banyakdata - 1]);

	enkripsiCUDA(m, k, g, p, y, res);

	// printf("<<<<<<<<<<<<<<Hasil Enkripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("c[%d] = %lu 	c[%d] = %lu\n", 2 * i, res[2 * i], 2 * i + 1, res[2 * i + 1]);
	// }

	// printf("c ...\n");
	// printf("c[%d] = %lu 	c[%d] = %lu\n", banyakdata * 2 - 2, res[banyakdata * 2 - 2], banyakdata * 2 - 1, res[banyakdata * 2 - 1]);

	e = p - x - 1;
	dekripsiCUDA(res, p, e, res2);

	// printf("<<<<<<<<<<<<<<Hasil Dekripsi>>>>>>>>>>>>>>>\n");
	// for (int i = 0; i < 4; i++) {
	// 	printf("m[%d] = %lu\n", i, res2[i]);
	// }

	// printf("m[...]\n");
	// printf("m[%d] = %lu\n", banyakdata - 1, res2[banyakdata - 1]);

	free(m);
	free(k);
	free(res);
	free(res2);

	return 0;
}