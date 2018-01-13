/*
 *
 * SOFT354
 * Ben Lancaster 10424877
 * Parallel DFT (Discrete Fourier Transform) algorithm
 * CUDA
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define M_PI 3.14159265358979323846
#include <time.h>
#include <string.h>
#include <windows.h>

#define _in_
#define _out_
#define _inout_

char *input_file_path = "../data/sine.large.csv";

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

TIME_DECL(time_alloc);
TIME_DECL(time_kernel);

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


int assert_vec(double *v, double *vt, int n)
{
        while (n--) {
                if (*v++ != *vt++) return 0;
        }

        return 1;
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
                        sumreal += x[n] * cos(n * k * 2 * M_PI / xn);
                        sumimag -= x[n] * sin(n * k * 2 * M_PI / xn);
                }

                (*fx)[k] = fabs(sumreal*sumreal) + fabs(sumimag*sumimag);
        }

        return 0;
}

__global__ void kernel_dft(int xn, double *a, double *q)
{
        int n;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Stop threads in block outside of xn
        if (idx > xn) return;

        double sum_real = 0;
        double sum_imag = 0;
        for (n = 0; n < xn; n++) {
                sum_real += a[n] * cos(n * idx * 2 * M_PI / xn);
                sum_imag -= a[n] * sin(n * idx * 2 * M_PI / xn);
        }

        // Write result to output vector
        q[idx] = fabs(sum_real*sum_real) + fabs(sum_imag * sum_imag);
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
        TIME_START(time_alloc);
        cuda_status = cudaMalloc(&dev_x, xn * sizeof(double));
        cuda_status = cudaMalloc(&dev_q, xn * sizeof(double));
        TIME_STOP(time_alloc);

        // Copy input vectors from host memory to GPU buffers.
        cuda_status = cudaMemcpy(dev_x, x, xn * sizeof(double), cudaMemcpyHostToDevice);

        TIME_START(time_kernel);
        kernel_dft <<<num_blocks, block_size>>>(xn, dev_x, dev_q);
        cudaDeviceSynchronize();
        TIME_STOP(time_kernel);

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

        read_into_v(input_file_path, &vt, &samples);
        read_into_v(input_file_path, &sf, &samples);

        out_dft = fopen("../test/dft.txt", "w");
        if (!out_dft) {
                printf("Error creating output dft file!\r\n");
                goto exit;
        }

        seq_dft(vt, samples, &sf);
        //print_vec(out_dft, sf, samples);

        cu_dft(vt, samples, &vf);
        fprint_vec(out_dft, sf, samples);

        printf("assert_vec: %s\r\n", 
                assert_vec(sf, vf, samples)? "SUCCESS" : "FAIL");

        for (i = 0; i < samples; i++) {
                printf("%d\t%f\t%f\t%s\r\n", i, sf[i], vf[i],
                        sf[i] == vf[i] ? "" : "!!!");
        }

exit:
        free(vt);
        free(vf);

        return 0;
}
