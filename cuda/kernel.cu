/*
 *
 * SOFT354
 * Ben Lancaster 10424877
 * Parallel DFT (Discrete Fourier Transform) algorithm
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define _in_
#define _out_
#define _inout_

int assert_vec(double *v, double *vt, int n)
{
        while (n--) {
                if (*v++ != *vt++) return 0;
        }

        return 1;
}

void print_vec(FILE *f, double *v, int n)
{
        int i;
        FILE *f_out;

        if (f) f_out = f;
        else f_out = stdout;
        for (i = 0; i < n; i++) {
                fprintf(f_out, "%d,%.2lf\n", i, v[i]);
        }
}

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

// Sequential implementation of the DFT algorithm
int seq_dft(
        _in_ double *x, _in_ int xn,
        _out_ double **fx)
{
        int k, n;
        //printf("Performing dft on input:\r\n\t");

        *fx = (double*)calloc(xn, sizeof(double));
        double *xr = (double*)calloc(xn, sizeof(double));
        double *xi = (double*)calloc(xn, sizeof(double));

        for (k = 0; k < xn; k++) {
                double sumreal = 0;
                double sumimag = 0;

                for (n = 0; n < xn; n++) {
                        sumreal += x[n] * cos(n * k * 2 * acos(-1) / xn);
                        sumimag -= x[n] * sin(n * k * 2 * acos(-1) / xn);
                }

                (*fx)[k] = abs(sumreal*sumreal) + abs(sumimag*sumimag);
        }

        return 0;
}

__global__ void kernel_dft(int xn, double *a, double *q)
{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx > xn) return;

        printf("%d\r\n", idx);
        q[idx] = a[idx];
}

cudaError_t cu_dft(
        _in_ double *x, _in_ int xn,
        _out_ double **fx)
{
        double *dev_x = NULL;
        double *dev_q = NULL;
        cudaError_t cuda_status;

        int block_size = 512;
        int num_blocks = (xn + block_size - 1) / block_size;
        printf("block_size: %d num_blocks: %d xn: %d\r\n",
                block_size, num_blocks, xn);

        // Allocate host output memory
        *fx = (double*)calloc(xn, sizeof(double));

        // Allocate GPU buffers for two vectors (one input, one output)
        cuda_status = cudaMalloc(&dev_x, xn * sizeof(double));
        cuda_status = cudaMalloc(&dev_q, xn * sizeof(double));

        // Copy input vectors from host memory to GPU buffers.
        cuda_status = cudaMemcpy(dev_x, x, xn * sizeof(double), cudaMemcpyHostToDevice);

        kernel_dft <<<num_blocks, block_size>>>(xn, dev_x, dev_q);

        // Check for any errors launching the kernel
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
                printf("addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
                goto error;
        }
        
        // Copy output vector from GPU buffer to host memory.
        cuda_status = cudaMemcpy(*fx, dev_q, xn * sizeof(double), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
                printf("cudaMemcpy failed!");
                goto error;
        }

error:
        cudaFree(dev_x);
        cudaFree(dev_q);

        return cuda_status;
}

int main()
{
        FILE    *out_dft; // Output file

        // Hyper parameters
        double  *sf, // sequentual dft f(x)
                *vt, // value 
                *vf; // cuda dft f(x)
        int     samples;
        int     i;

        read_into_v("../data/square.csv", &vt, &samples);
        read_into_v("../data/square.csv", &sf, &samples);

        out_dft = fopen("../test/dft.txt", "w");
        if (!out_dft) {
                printf("Error creating output dft file!\r\n");
                goto exit;
        }

        seq_dft(vt, samples, &sf);
        print_vec(out_dft, sf, samples);

        cu_dft(vt, samples, &vf);

        printf("assert_vec: %s\r\n", 
                assert_vec(vt, vf, samples)? "SUCCESS" : "FAIL");
/*
        for (i = 0; i < samples; i++) {
                printf("%d\t%f\t%f\t%s\r\n", i, vt[i], vf[i],
                        vt[i] == vf[i] ? "" : "!!!");
        }*/

exit:
        free(vt);
        free(vf);

        return 0;
}
