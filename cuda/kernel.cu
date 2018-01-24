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

// Input data set
//char *input_file_path = "../data/sine.10000.csv";
char *input_file_path = "../data/sine.large.csv";
//char *input_file_path = "../data/sine.csv";

// Toggle building cuda kernel with shared memory
#define USE_CUDA_SHARED 0
__shared__ double a_shared[];

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

// Add things to time
TIME_DECL(time_alloc);
TIME_DECL(time_kernel);
TIME_DECL(time_total);

// Helper functions
/// Returns number of lines in file <path>
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

/// Reads csv file <path> into vector *v
/// Populates number of samples in *vn
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

// Prints an array to screen
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

// Tests if two arrays are equal
int assert_vec(double *v, double *vt, int n)
{
        while (n--) {
                if (*v++ != *vt++) return 0;
        }

        return 1;
}

// Sequential implementation of the DFT algorithm for comparison
// https://www.nayuki.io/page/how-to-implement-the-discrete-fourier-transform
int seq_dft(
        _in_ double *x, _in_ int xn,
        _out_ double **fx)
{
        int k, n;
        const double phi = 2 * M_PI / xn;
        //printf("Performing dft on input:\r\n\t");

        *fx = (double*)calloc(xn, sizeof(double));
        double *xr = (double*)calloc(xn, sizeof(double));
        double *xi = (double*)calloc(xn, sizeof(double));

        for (k = 0; k < xn; k++) {
                double sum_real = 0;
                double sum_imag = 0;

                for (n = 0; n < xn; n++) {
                        sum_real += x[n] * cos(n * k * phi);
                        sum_imag -= x[n] * sin(n * k * phi);
                }

                (*fx)[k] = fabs(sum_real*sum_real) + fabs(sum_imag*sum_imag);
        }

        return 0;
}

// CUDA kernel for DFT
// Each sample is assigned a thread
__global__ void kernel_dft(
        int xn,                 // Number of samples
        double *a,              // Input time vector
        double *q,              // Output frequency vector
        int block_size          // Block x dimension
){
        int n;
#if USE_CUDA_SHARED == 1
        extern __shared__ double a_shared[];
#endif
        // 1D grid and block dimensions
        // Sample index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const double phi = 2 * M_PI / xn;
        
        // Stop threads in block outside of xn
        if (idx > xn) return;

#if USE_CUDA_SHARED == 1
        // First thread of each block must copy global memory to shared
        if (idx % block_size == 0) {
                memcpy(a_shared, a, xn * sizeof(double));
        }
        // Sync all threads for shared mem to be ready
        __syncthreads();
#endif

        // Do the DFT for this sample[idx]
        double sum_real = 0;
        double sum_imag = 0;
        for (n = 0; n < xn; n++) {
#if USE_CUDA_SHARED == 1
                sum_real += a_shared[n] * cos(n * idx * phi);
                sum_imag -= a_shared[n] * sin(n * idx * phi);
#else
                sum_real += a[n] * cos(n * idx * phi);
                sum_imag -= a[n] * sin(n * idx * phi);
#endif
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

        int block_size = 64;
        int num_blocks = (xn + block_size - 1) / block_size;
        printf("block_size: %d num_blocks: %d xn: %d\r\n",
                block_size, num_blocks, xn);

        // Allocate host output memory
        *fx = (double*)calloc(xn, sizeof(double));

        TIME_START(time_total);

        // Allocate GPU buffers for two vectors (one input, one output)
        TIME_START(time_alloc);
        cuda_status = cudaMalloc(&dev_x, xn * sizeof(double));
        cuda_status = cudaMalloc(&dev_q, xn * sizeof(double));
        TIME_STOP(time_alloc);

        // Copy input vectors from host memory to GPU buffers.
        cuda_status = cudaMemcpy(dev_x, x, xn * sizeof(double), cudaMemcpyHostToDevice);

        TIME_START(time_kernel);
        kernel_dft <<< 
                num_blocks,             // grid dimensions
                block_size,             // block dimensions
                xn * sizeof(double)     // dynamic shared memory size
                >>> (xn, dev_x, dev_q, block_size);
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

        TIME_STOP(time_total);

error:
        cudaFree(dev_x);
        cudaFree(dev_q);

        return cuda_status;
}

int main()
{
        // Output file
        FILE    *out_dft;

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
