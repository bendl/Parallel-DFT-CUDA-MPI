/*
 *
 * SOFT354
 * Ben Lancaster 10424877
 * Parallel DFT (Discrete Fourier Transform) algorithm
 * MPI
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <windows.h>

#include <mpi.h>

#define M_PI 3.14159265358979323846
#define _in_
#define _out_
#define _inout_
#define SAFE_FREE(x) if ((x)) free((x))

int     world_rank;
int     world_size;

int     nsamples;
int     nsamples_per_node;
int     nsamples_start;

char    *input_file_path = "../data/sine.large.csv";

FILE    *out_dft; // Output file

#define ROOT_RANK       (0)
#define IS_ROOT         (world_rank == ROOT_RANK)
#define ROOT_ONLY       if (IS_ROOT)
#define NOT_ROOT        if (!IS_ROOT)



// Time recording
double  timer_freq = 0;
#define TIME_DECL(name) \
        __int64 name##_start; \
        double  name

#define TIME_START(d) \
        timer_start(&##d##_start, &timer_freq);

#define __TIME_STOP(d) \
        d = timer_stop(&##d##_start, &timer_freq);

#define TIME_STOP(d) \
        __TIME_STOP(d) \
        printf("\t" # d "\t%f ms\r\n", d);

#define TIME_STOP_R(d) \
        __TIME_STOP(d) \
        ROOT_ONLY { printf("\t" # d "\t%f ms\r\n", d); }

TIME_DECL(time_gather);
TIME_DECL(time_bcast);
TIME_DECL(time_total);
TIME_DECL(time_dft);
TIME_DECL(time_seq_dft);

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



// Sequential implementation of the DFT algorithm
int seq_dft(
        _in_ double *x, _in_ int xn,
        _out_ double **fx)
{
        int k, n;
        //printf("Performing dft on input:\r\n\t");

        // Allocate memory for output vector function
        *fx = (double*)calloc(xn, sizeof(double));

        for (k = 0; k < xn; k++) {
                double sumreal = 0;
                double sumimag = 0;

                for (n = 0; n < xn; n++) {
                        sumreal += x[n] * cos(n * k * 2 * M_PI / xn);
                        sumimag -= x[n] * sin(n * k * 2 * M_PI / xn);
                }

                (*fx)[k] = fabs(sumreal*sumreal) + fabs(sumimag*sumimag);
        }

        return 0;
}

// Parallel implementation of DFT algorithm in MPI
int mpi_dft(
        _in_ double *vt,        // vector time domain
        _in_ int n_start,       // f(n_start) location
        _in_ int N,             // vector time domain count
        _out_ double **fx       // vector frequency domain out
) {
        int k, n;

        if (n_start > nsamples) return 1;
        printf("#%d nstart: %d\r\n", world_rank, n_start);

        // Allocate memory for output vector function
        *fx = (double*)calloc(nsamples_per_node, sizeof(double));

        // Perform the dft starting from this sample
        for (k = 0; k < nsamples_per_node; k++) {
                double  sum_real = 0;
                double  sum_imag = 0;
                double  sum_out = 0;
                int     sample_start;

                // Out of bounds check
                if ((k + nsamples_start) > nsamples/2) break;

                for (n = 0; n < N; n++) {
                        sum_real += vt[n] * cos(n * (k + nsamples_start) * 2 * M_PI / N);
                        sum_imag -= vt[n] * sin(n * (k + nsamples_start) * 2 * M_PI / N);
                }

                sum_out = fabs(sum_real*sum_real) + fabs(sum_imag*sum_imag);
                //printf("#%d writing %d: %f\r\n", world_rank, k, sum_out);
                (*fx)[k] = fabs(sum_real*sum_real) + fabs(sum_imag*sum_imag);
        }

        return 0;
}

int main(int argc, char **argv)
{
        int ret = 0;

        // Hyper parameters
        double  *sf, // sequentual dft f(x)
                *vt, // value in time domain
                *vf, // value in frequency domain
                *vf_all = NULL;
        int     i;

        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ROOT_ONLY {
                read_into_v(input_file_path, &vt, &nsamples);
                read_into_v(input_file_path, &sf, &nsamples);

                // Create output file
                out_dft = fopen("../test/dft.txt", "w");
                if (!out_dft) {
                        ret = 1;
                        goto exit;
                }
        }

#if 1
        ROOT_ONLY {
                TIME_START(time_seq_dft);
                seq_dft(vt, nsamples, &vf);
                TIME_STOP(time_seq_dft);
        }
#endif

        // Total program run time timer
        TIME_START(time_total);

        // Broadcast the nsamples to each node
        TIME_START(time_bcast);
        MPI_Bcast(&nsamples, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
        TIME_STOP(time_bcast);

        // Allocate memory for the incoming sample buffer
        NOT_ROOT { 
                vt = malloc(nsamples * sizeof(double)); 
        }

        // Broadcast the samples array to each node.
        // Each node will need the full samples array as it needs to sum
        // each value individually
        MPI_Bcast(vt, nsamples, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

        // Calculate how many iterations each node should perform
        nsamples_per_node = (nsamples + world_size - 1) / world_size;
        nsamples_start = world_rank * nsamples_per_node;

        /*
        printf("%d/%d nsamples: %d Per node: %d Starting: %d\r\n", 
                world_rank, world_size, nsamples, nsamples_per_node, nsamples_start);
        printf("Rank %d: First samples: %f %f %f %f\r\n", 
                world_rank, vt[0], vt[1], vt[2], vt[3]);
        //*/

        if (nsamples_start > nsamples) {
                // More nodes than datapoints
                ret = 1;
                goto exit;
        }

        ROOT_ONLY {
                printf("Samples Per Node: %d\r\n", nsamples_per_node);
                // Combined output of all nodes
                // Note: Size is not nsamples, as the last block may not
                // have same number of samples as the others
                vf_all = calloc(world_size * nsamples_per_node, sizeof(double));
        }

        // Perform the parallel dft function
        TIME_START(time_dft);
        mpi_dft(vt, nsamples_start, nsamples, &vf);
        TIME_STOP_R(time_dft);

        // Now ROOT_RANK must gather all sub arrays into a single array
        TIME_START(time_gather);
        MPI_Gather(
                vf,     nsamples_per_node, MPI_DOUBLE, // send
                vf_all, nsamples_per_node, MPI_DOUBLE, // recv
                ROOT_RANK, MPI_COMM_WORLD);
        TIME_STOP_R(time_gather);

        TIME_STOP_R(time_total);

        ROOT_ONLY {
                // Write output function
                fprint_vec(out_dft, vf_all, nsamples);
        }

exit:
        MPI_Finalize();
        SAFE_FREE(vt);
        SAFE_FREE(vf);

        ROOT_ONLY {
                SAFE_FREE(sf);
                SAFE_FREE(vf_all);
                fclose(out_dft);
        }

        return ret;
}
