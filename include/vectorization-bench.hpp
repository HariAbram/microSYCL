#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

using namespace cl;

void gemv_range_usm(sycl::queue &Q, int size);

void gemv_range_buff_acc(sycl::queue &Q, int size);

void gemv_ndrange_usm(sycl::queue &Q, int size, int block_size);

void gemv_ndrange_buff_acc(sycl::queue &Q, int size, int block_size);

void gemm_range_usm(sycl::queue &Q, int size);

void gemm_range_buff_acc(sycl::queue &Q, int size);

void gemm_ndrange_usm(sycl::queue &Q, int size, int block_size);

void gemm_ndrange_buff_acc(sycl::queue &Q, int size, int block_size);

void triad(sycl::queue &Q, int size, int block_size);

void outer_product(sycl::queue &Q, int size, int block_size);

void cross_product(sycl::queue &Q, int size, int block_size);




