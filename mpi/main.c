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

#include <mpi.h>

#include "../util/soft354_file.h"

int main(int argc, char **argv)
{
        printf("Hello world\r\n");

        MPI_Init(argc, argv);

        FILE    *out_dft; // Output file

                          // Hyper parameters
        double  *sf, // sequentual dft f(x)
                *vt, // value 
                *vf; // cuda dft f(x)
        int     samples;
        int     i;

        read_into_v("../data/square.csv", &vt, &samples);
        read_into_v("../data/square.csv", &sf, &samples);

exit:
        MPI_Finalize();

        return 0;
}