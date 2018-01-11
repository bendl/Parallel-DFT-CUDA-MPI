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

#include <mpi.h>

#include "../util/soft354_file.h"

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

FILE    *out_dft; // Output file

#define ROOT_RANK       (0)
#define IS_ROOT         (world_rank == ROOT_RANK)
#define ROOT_ONLY       if (IS_ROOT)
#define NOT_ROOT        if (!IS_ROOT)

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

// Sequential implementation of the DFT algorithm
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
                double sumreal = 0;
                double sumimag = 0;
                double sum_out = 0;
                int sample_start;

                // Out of bounds check
                if ((k + nsamples_start) > nsamples/2) break;

                for (n = 0; n < N; n++) {
                        sumreal += vt[n] * cos(n * (k + nsamples_start) * 2 * M_PI / N);
                        sumimag -= vt[n] * sin(n * (k + nsamples_start) * 2 * M_PI / N);
                }

                sum_out = fabs(sumreal*sumreal) + fabs(sumimag*sumimag);
                //printf("#%d writing %d: %f\r\n", world_rank, k, sum_out);
                (*fx)[k] = fabs(sumreal*sumreal) + fabs(sumimag*sumimag);
        }

        return 0;
}

int main(int argc, char **argv)
{
        int ret = 0;

        // Hyper parameters
        double  *sf, // sequentual dft f(x)
                *vt, // value 
                *vf, // cuda dft f(x) (SUB SAMPLE)
                *vf_all = NULL;
        int     i;

        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ROOT_ONLY {
                read_into_v("../data/sine.csv", &vt, &nsamples);
                read_into_v("../data/sine.csv", &sf, &nsamples);

                // Create output file
                out_dft = fopen("../test/dft.txt", "w");
                if (!out_dft) {
                        ret = 1;
                        goto exit;
                }
        }

        // Broadcast the nsamples to each node
        MPI_Bcast(&nsamples, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

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
        mpi_dft(vt, nsamples_start, nsamples, &vf);

        // Now ROOT_RANK must gather all sub arrays into a single array
        MPI_Gather(
                vf,     nsamples_per_node, MPI_DOUBLE, // send
                vf_all, nsamples_per_node, MPI_DOUBLE, // recv
                ROOT_RANK, MPI_COMM_WORLD);

        ROOT_ONLY {
                // Write output function
                print_vec(out_dft, vf_all, nsamples);
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
