#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

//using namespace cl;

double delay_time();

// memory allocation 

void memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter);

void host_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter);

void shared_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter);

void device_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter);

//parallelization

void range_with_usm(sycl::queue &Q, int size, int dim, bool print, int iter);

void range_with_buff_acc(sycl::queue &Q, int size, int dim, bool print, int iter);

void nd_range_with_usm(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter);

void nd_range_with_buff_acc(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter);

//reduction

void atomics_buf_acc(sycl::queue &Q, int size, bool print, int iter);

void atomics_usm(sycl::queue &Q, int size, bool print, int iter);

void reduction_with_buf_acc(sycl::queue &Q, int size, int block_size, bool print, int iter);

void reduction_with_usm(sycl::queue &Q, int size, int block_size, bool print, int iter);

// barriers

void group_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim);

void group_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim);

void subgroup_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim);

void subgroup_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter, int dim);



