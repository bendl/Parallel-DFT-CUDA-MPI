/*
*
* SOFT354
* Ben Lancaster 10424877
* Parallel DFT (Discrete Fourier Transform) algorithm
* File IO
*
*/

#include "soft354_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

int read_get_lines(char *path)
{
        FILE *f;
        int ch, nlines = 0;

        f = fopen(path, "r");
        if (!f) {
                return 0;
        }

        while ((ch = fgetc(f)) != EOF) {
                if (ch == '\n') {
                        nlines++;
                }
        }

        fclose(f);

        return nlines;
}

int read_into_v(char *path, double **v, int *vn)
{
        FILE *f;
        char line_buf[256];
        char *lp;

        int line = 0;
        int col = 0;
        int i = 0;
        int j = 0;

        *vn = read_get_lines(path);
        *v = (double*)calloc(*vn, sizeof(double));

        f = fopen(path, "r");
        if (!f) {
                return 0;
        }

        while (fgets(line_buf, 256, f)) {
                if (++line > 2) {
                        lp = strtok(line_buf, ",");
                        while (lp != NULL) {
                                if (col == 1) {
                                        double val = -1;
                                        //printf("%s\r\n", lp);
                                        if (sscanf(lp, "%lf", &val) == 1) {
                                                (*v)[i++] = val;
                                        }
                                }
                                lp = strtok(NULL, ",");
                                col++;
                        }
                        col = 0;
                }
        }
}

void fprint_vec(FILE *f, double *v, int n)
{
        int i;
        FILE *f_out;

        if (f) f_out = f;
        else f_out = stdout;
        for (i = 0; i < n; i++) {
                fprintf(f_out, "%d,%.2lf\n", i, v[i]);
        }
}

void timer_start(__int64 * start, double *freq)
{
        LARGE_INTEGER li;
        
        if (!QueryPerformanceFrequency(&li)) return;

        *freq = (double)(li.QuadPart / 1000.0);

        QueryPerformanceCounter(&li);
        *start = li.QuadPart;
}

double timer_stop(__int64 * start, double *freq)
{
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return (double)(li.QuadPart - *start) / *freq;
}

