/*
 * SOFT354
 * Ben Lancaster 10424877
 * DFT Algorithm MPI
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
        printf("Hello world\r\n");

        MPI_Init(argc, argv);

exit:
        MPI_Finalize();

        return 0;
}