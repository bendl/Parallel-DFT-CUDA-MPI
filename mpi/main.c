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

int main(int argc, char **argv)
{
        int ret = 0;

        // Hyper parameters
        double  *sf, // sequentual dft f(x)
                *vt, // value 
                *vf; // cuda dft f(x)
        int     nsamples;
        int     i;

        MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ROOT_ONLY {
                read_into_v("../data/square.csv", &vt, &nsamples);
                read_into_v("../data/square.csv", &sf, &nsamples);

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

        printf("%d/%d nsamples: %d Per node: %d Starting: %d\r\n", 
                world_rank, world_size, nsamples, nsamples_per_node, nsamples_start);
        printf("Rank %d: First samples: %f %f %f %f\r\n", 
                world_rank, vt[0], vt[1], vt[2], vt[3]);

        ROOT_ONLY {
                seq_dft(vt, nsamples, &vf);
                print_vec(out_dft, vf, nsamples);
        }

exit:
        MPI_Finalize();
        SAFE_FREE(vt);

        ROOT_ONLY {
                SAFE_FREE(sf);
        }

        return ret;
}