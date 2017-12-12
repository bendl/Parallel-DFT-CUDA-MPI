/*
*
* SOFT354
* Ben Lancaster 10424877
* Parallel DFT (Discrete Fourier Transform) algorithm
* File IO
*
*/

#ifndef SOFT354_FILE_H
#define SOFT354_FILE

#include <stdio.h>

int read_get_lines(char *path);
int read_into_v(char *path, double **v, int *vn);

void print_vec(FILE *f, double *v, int n);

#endif
