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

void fprint_vec(FILE *f, double *v, int n);


void timer_start(__int64 * start, double *freq);
double timer_stop(__int64 * start, double *freq);

#endif
