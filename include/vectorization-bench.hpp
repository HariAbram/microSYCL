#ifndef VECTORIZATION_HPP
#define VECTORIZATION_HPP

#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

/*gemv*/
void gemv_range_usm(sycl::queue &Q, int size);

void gemv_range_buff_acc(sycl::queue &Q, int size);

void gemv_ndrange_usm(sycl::queue &Q, int size, int block_size);

void gemv_ndrange_buff_acc(sycl::queue &Q, int size, int block_size);

/*gemm*/
void gemm_range_usm(sycl::queue &Q, int size);

void gemm_range_buff_acc(sycl::queue &Q, int size);

void gemm_ndrange_usm(sycl::queue &Q, int size, int block_size);

void gemm_ndrange_buff_acc(sycl::queue &Q, int size, int block_size);

void gemm_opt_ndrange_usm(sycl::queue &Q, int size, int block_size);

/*triad*/
void triad(sycl::queue &Q, int size, int block_size);

/*outer-product*/
void outer_product(sycl::queue &Q, int size, int block_size);

/*cross-product*/
void cross_product(sycl::queue &Q, int size, int block_size);

/*Sparse*/

void spmv_csr_ndrange_usm(sycl::queue &Q, int size, int block_size);

//void spmm__ndrange_usm(sycl::queue &Q, int size, int block_size);

#endif
