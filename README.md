# Parallel DFT for the GPU and Software

This repository designs and evaluates a parallel DFT (Discrete Fourier Transform) implementation on CUDA GPUs and MPI (Message Passing Interface) architectures.

## Implementation
| CUDA  | MPI |
| ------------- | ------------- |
| ![CUDA Impl](https://raw.githubusercontent.com/bendl/soft354/master/report/img/cuda_impl1.jpg) | ![MPI Impl](https://raw.githubusercontent.com/bendl/soft354/master/report/img/mpi_impl1.jpg) |

## Evaluation
Both CUDA and MPI implementations saw significant performance benefits over the sequential algorithm.

![Performance Graph showing better parallel performance](https://raw.githubusercontent.com/bendl/soft354/master/report/img/seq_vs_par.JPG)
